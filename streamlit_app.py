import streamlit as st
import openai
import pandas as pd
import base64
import json
import re
import imapclient
import pyzmail
import os
import ast
import pdfplumber
import io

# OpenAI API Key
OPENAI_API_KEY = "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"  # Replace with your key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize session state for projects if not already initialized
if "projects" not in st.session_state:
    st.session_state.projects = {}  # Dictionary to store project data

# Function to process invoice using OpenAI API
def process_invoice(image_data):
    """Extract structured data from an invoice image using GPT-4o."""
    try:
        base64_image = base64.b64encode(image_data).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that extracts structured data from invoices."},
                {"role": "user", "content": [
                    {"type": "text", "text": (
                        "Extract the following details from this invoice and return them in JSON format:\n"
                        "- Invoice Number\n- Invoice Date\n- Supplier Name\n- Supplier VAT\n"
                        "- Customer Name\n- Customer VAT\n- Total Amount\n- VAT Amount"
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=500
        )

        # Extract and clean response
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()

        # Convert response to dictionary
        invoice_data = json.loads(response_text)
        return invoice_data

    except json.JSONDecodeError:
        st.error("Error: Unable to decode JSON response.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    return None  # Return None if an error occurs

def extract_text_from_pdf(pdf_bytes):
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text

def extract_resume_data_with_gpt(text):
    prompt = f"""
You are a resume parser. Extract the following details from the resume below and return them as a JSON object:
- Full Name
- Email
- Phone Number
- Skills (comma-separated)
- Education (Degree + Institute)
- Total Experience (in years)
- Last Company
- Last Designation

Resume Text:
{text[:3500]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract structured info from resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        print("Response:", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error:", e)
        return None
def clean_and_parse_gpt_json(response_text):
    # Remove markdown formatting if present
    cleaned = re.sub(r"```json|```", "", response_text).strip()
    return json.loads(cleaned)

def parse_resumes_from_email(subject_keyword, email_user, email_pass, imap_server='mail.jffconsultants.com'):
    import imapclient, pyzmail, pandas as pd, ast

    results = []

    with imapclient.IMAPClient(imap_server) as imap:
        imap.login(email_user, email_pass)
        imap.select_folder('INBOX', readonly=True)

        messages = imap.search(['SUBJECT', subject_keyword])
        for uid in messages:
            raw = imap.fetch([uid], ['BODY[]'])
            message = pyzmail.PyzMessage.factory(raw[uid][b'BODY[]'])
            print("Processing message:", message.get_subject())

            for part in message.mailparts:
                if part.filename and part.filename.lower().endswith('.pdf'):
                    pdf_data = part.get_payload()
                    text = extract_text_from_pdf(pdf_data)
                    resume_str = extract_resume_data_with_gpt(text)

                    if resume_str:
                        try:
                            parsed = clean_and_parse_gpt_json(resume_str)
                            results.append(parsed)
                        except Exception as e:
                            print("Parsing error:", e)

    df = pd.DataFrame(results)

    # Deduplicate based on Email (preferred) or Full Name
    if 'Email' in df.columns:
        df.drop_duplicates(subset='Email', inplace=True)
    elif 'Full Name' in df.columns:
        df.drop_duplicates(subset='Full Name', inplace=True)

    return df.reset_index(drop=True)
def match_resumes_to_job(df, job_desc, client):
    results = []

    for idx, row in df.iterrows():
        resume_text = f"""
Name: {row.get('Full Name', '')}
Email: {row.get('Email', '')}
Phone: {row.get('Phone Number', '')}
Skills: {row.get('Skills', '')}
Education: {row.get('Education', '')}
Experience: {row.get('Total Experience', '')} years
Last Company: {row.get('Last Company', '')}
Last Designation: {row.get('Last Designation', '')}
"""

        prompt = f"""
You are an HR assistant. Compare the candidate's resume against the following job description and give a match score (out of 100) and a short reasoning (max 3 lines).

Job Description:
{job_desc}

Resume:
{resume_text}

Respond in this JSON format:
{{
  "Score": int,
  "Reason": "short text"
}}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You score resumes against job descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            result = response.choices[0].message.content.strip()
            parsed = json.loads(re.sub(r"```json|```", "", result).strip())
            row_data = row.to_dict()
            row_data.update({
                "Match Score": parsed["Score"],
                "Match Reason": parsed["Reason"]
            })
            results.append(row_data)

        except Exception as e:
            print(f"Matching error for {row.get('Full Name', '')}: {e}")
            continue

    # Return top 5 candidates by score
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="Match Score", ascending=False).head(5)
    return result_df.reset_index(drop=True)

# Streamlit App Layout
st.title("Invoice Processing App")
page = st.sidebar.radio("Go to", ["Invoice Extraction", "Extract Resumes"])
if page == "Invoice Extraction":

    # Sidebar: Add New Project
    st.sidebar.header("Project Management")
    project_name = st.sidebar.text_input("Enter New Project Name")

    if st.sidebar.button("Add Project"):
        if project_name:
            if project_name not in st.session_state.projects:
                st.session_state.projects[project_name] = []  # Initialize project with empty invoice list
                st.sidebar.success(f"Project '{project_name}' added!")
            else:
                st.sidebar.warning("Project already exists!")
        else:
            st.sidebar.error("Project name cannot be empty.")

    # Select Project Dropdown
    selected_project = st.sidebar.selectbox("Select a Project", list(st.session_state.projects.keys()))

    if selected_project:
        st.subheader(f"Project: {selected_project}")

        # Upload multiple invoices
        uploaded_files = st.file_uploader("Upload Invoices (PNG, JPG)", type=["png", "jpg"], accept_multiple_files=True)

        if st.button("Process Invoices"):
            if uploaded_files:
                existing_invoices = {inv["Invoice Number"] for inv in st.session_state.projects[selected_project]}  # Track existing invoice numbers
                new_invoices = []
                repeated_invoices = []

                for uploaded_file in uploaded_files:
                    # Read file as binary
                    file_data = uploaded_file.read()

                    # Process invoice
                    invoice_data = process_invoice(file_data)

                    if invoice_data:
                        invoice_number = invoice_data.get("Invoice Number")

                        if invoice_number in existing_invoices:
                            repeated_invoices.append(invoice_number)
                        else:
                            invoice_data["File Name"] = uploaded_file.name  # Track file name
                            new_invoices.append(invoice_data)
                            existing_invoices.add(invoice_number)  # Update existing invoices set

                # Save new invoices
                st.session_state.projects[selected_project].extend(new_invoices)

                # Success message
                if new_invoices:
                    st.success(f"Processed {len(new_invoices)} new invoice(s) successfully!")

                # Warning for duplicates
                if repeated_invoices:
                    st.warning(f"Skipped {len(repeated_invoices)} repeated invoice(s): {', '.join(repeated_invoices)}")

        # Display Invoices in a Table
        if st.session_state.projects[selected_project]:
            df = pd.DataFrame(st.session_state.projects[selected_project])

            # Convert amounts to numeric for summation
            df["Total Amount"] = pd.to_numeric(df["Total Amount"], errors="coerce")
            df["VAT Amount"] = pd.to_numeric(df["VAT Amount"], errors="coerce")

            st.dataframe(df)

            # Display Total Amount and Total VAT at the bottom
            total_amount = df["Total Amount"].sum()
            total_vat = df["VAT Amount"].sum()

            st.markdown(f"### **Total Amount: {total_amount:.2f}**")
            st.markdown(f"### **Total VAT: {total_vat:.2f}**")

elif page == "Extract Resumes":
    st.title("AI Resume Parser")

    subject = st.text_input("Enter subject keyword to fetch resumes")
    df = pd.DataFrame()  # Initialize empty DataFrame for resumes
    if "resumes_df" not in st.session_state:
        st.session_state.resumes_df = pd.DataFrame()
    if "top_candidates_df" not in st.session_state:
        st.session_state.top_candidates_df = pd.DataFrame()

    if st.button("Fetch & Parse Resumes"):
        with st.spinner("Processing resumes..."):
            EMAIL_USER = "careers@jffconsultants.com"
            EMAIL_PASS = "f3HYwGp^FQ0Z"  # Replace with your email password
            df = parse_resumes_from_email(subject, EMAIL_USER, EMAIL_PASS)
            st.session_state.resumes_df = df
            st.success("Parsing complete!")
            st.dataframe(df)

            # Optional: Add download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "resumes.csv", "text/csv")
    st.subheader("Job Description Matching")

    job_desc = st.text_area("Enter brief job description", height=200)

    if st.button("Find Top 5 Candidates"):
        if not job_desc:
            st.warning("Please enter a job description.")
        elif st.session_state.resumes_df.empty:
            st.warning("No resumes found.")
        else:
            with st.spinner("Matching resumes..."):
                top_candidates = match_resumes_to_job(st.session_state.resumes_df, job_desc, client)
                st.session_state.top_candidates_df = top_candidates
                st.success("Top matches found!")
    
    if not st.session_state.resumes_df.empty:
        st.subheader("All Parsed Resumes")
        st.dataframe(st.session_state.resumes_df)
        csv_all = st.session_state.resumes_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download All Resumes", csv_all, "resumes.csv", "text/csv")

    if not st.session_state.top_candidates_df.empty:
        st.subheader("Top 5 Matched Candidates")
        st.dataframe(st.session_state.top_candidates_df)
        csv_top = st.session_state.top_candidates_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Top 5", csv_top, "top_candidates.csv", "text/csv")


