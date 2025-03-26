from flask import Flask, request, jsonify, send_from_directory
import pdfplumber
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def normalize_table(table):
    header = table[0]
    normalized = [header]
    for row in table[1:]:
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[:len(header)]
        normalized.append(row)
    return normalized


def extract_groww_tables(pdf_path, password):
    statement_data = []
    holding_data = []
    current_section = None

    with pdfplumber.open(pdf_path, password=password) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            if "statement of transaction" in text.lower():
                current_section = "statement"
            elif "holding balance" in text.lower() or "holdings balance" in text.lower():
                current_section = "holding"

            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                if current_section == "statement":
                    if not statement_data:
                        statement_data = table
                    else:
                        statement_data.extend(table[1:])
                elif current_section == "holding":
                    if "ISIN Code" in table[0]:
                        holding_data = table

    return normalize_table(statement_data), normalize_table(holding_data)


@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files or 'pan' not in request.form:
        return jsonify({"error": "Missing file or PAN input"}), 400

    file = request.files['file']
    pan = request.form['pan'].strip().upper()

    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        statement, holding = extract_groww_tables(filepath, pan)
        response = {"message": "Data extracted successfully"}

        if statement:
            df_statement = pd.DataFrame(statement[1:], columns=statement[0])
            response["transactions"] = df_statement.to_dict(orient="records")
        if holding:
            df_holdings = pd.DataFrame(holding[1:], columns=holding[0])
            response["holdings"] = df_holdings.to_dict(orient="records")

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
