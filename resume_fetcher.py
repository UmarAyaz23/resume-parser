import requests
import os
import concurrent
import streamlit as st
import pandas as pd
import json
import re
import imapclient
import pyzmail
import ast
import pdfplumber
import io
import groq
from dotenv import load_dotenv


load_dotenv()
client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_text_from_PDF(pdfbytes):
    with pdfplumber.open(io.BytesIO(pdfbytes)) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text


def extract_content_with_LLM(text, timeout = 15):
    prompt = f"""
        You are a resume parser. Extract the following details from the resume below and return ONLY a valid JSON object, no other text or explanation. Fields:

        - Full Name
        - Email
        - Phone Number
        - Skills (comma-separated string)
        - Education (Degree + Institute)
        - Total Experience (number of years)
        - Last Company
        - Last Designation

        Only output a JSON like this:
        {{
        "Full Name": "John Doe",
        "Email": "john@example.com",
        "Phone Number": "123-456-7890",
        "Skills": "Python, Java, SQL",
        "Education": "BS Computer Science - XYZ University",
        "Total Experience": "3",
        "Last Company": "ABC Corp",
        "Last Designation": "Software Engineer"
        }}

        Resume Text:
        {text[:3500]}
    """

    def _call_llm():
        return client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You extract structured information from resumes"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_call_llm)
            response = future.result(timeout=timeout)
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM Timeout/Error] Skipping this PDF: {e}")
        return None
    

def clean_parse_llm_json(response_text):
    cleaned = re.sub(r"```json|```", "", response_text).strip()
    return json.loads(cleaned)


def parse_resumes_from_email(keyword, user_email, email_password, imap_server = "mail.jffconsultants.com"):
    results = []

    with imapclient.IMAPClient(imap_server) as imap:
        imap.login(user_email, email_password)
        imap.select_folder('INBOX', readonly = True)

        messages = imap.search(['SUBJECT', keyword])
        for uid in messages[:8]:
            raw = imap.fetch([uid], ['BODY[]'])
            message = pyzmail.PyzMessage.factory(raw[uid][b'BODY[]'])
            print("Processing message: ", message.get_subject())

            for part in message.mailparts:
                if part.filename and part.filename.lower().endswith('pdf'):
                    pdf_data = part.get_payload()
                    text = extract_text_from_PDF(pdf_data)
                    resume_str = extract_content_with_LLM(text)

                    if resume_str:
                        try:
                            parsed = clean_parse_llm_json(resume_str)
                            results.append(parsed)
                        except Exception as e:
                            print(f"[Parsing Error] Invalid JSON or bad structure: {e}")
                    else:
                        print("[Warning] LLM returned None for resume, skipping.")

    df = pd.DataFrame(results)

    if "Email" in df.columns:
        df.drop_duplicates(subset = "Email", inplace = True)
    elif "Full Name" in df.columns:
        df.drop_duplicates(subset = "Full Name", inplace = True)

    return df.reset_index(drop = True)


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
                model = "llama-3.1-8b-instant",
                messages = [
                    {"role": "system", "content": "You score resumes against job descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature = 0.3,
                max_tokens = 300
            )

            result = response.choices[0].message.content
            cleaned_result = re.sub(r"```json|```", "", result).strip()
            parsed = json.loads(cleaned_result)
            row_data = row.to_dict()
            row_data.update({
                "Match Score": parsed["Score"],
                "Match Reason": parsed["Reason"]
            })
            results.append(row_data)
        
        except Exception as e:
            print(f"Matching error for {row.get('Full Name', '')}: {e}")
            continue

    if results:
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(by="Match Score", ascending=False).head(5)
        return result_df.reset_index(drop=True)
    else:
        return pd.DataFrame()
    

#---------------------------------------------------------------------STREAMLIT APP---------------------------------------------------------------------
st.title("Resume Parser")
page = st.sidebar.radio("Go To", ["Extract Resumes"])

if page == "Extract Resumes":
    st.title("AI Resume Parser")

    keyword = st.text_input("Enter a Keyword to Fetch Resumes")
    df = pd.DataFrame()

    if "resumes_df" not in st.session_state: 
        st.session_state.resumes_df = pd.DataFrame()

    if "top_candidates_df" not in st.session_state:
        st.session_state.top_candidates_df = pd.DataFrame()

    if st.button("Fetch & Parse Resumes"):
        with st.spinner("Processing..."):
            user_email = "careers@jffconsultants.com"
            email_password = "f3HYwGp^FQ0Z"

            df = parse_resumes_from_email(keyword, user_email, email_password)
            st.session_state.resumes_df = df
            st.success("Parsing Complete")


    if "resumes_df" in st.session_state and not st.session_state.resumes_df.empty:
        search_term = st.text_input("Search Resumes (Name, Email, Skills, etc.)")
        filtered_df = st.session_state.resumes_df

        if search_term:
            # Case-insensitive filtering across multiple columns
            search_term_lower = search_term.lower()

            mask = (
                filtered_df["Full Name"].str.lower().str.contains(search_term_lower, na=False) |
                filtered_df["Email"].str.lower().str.contains(search_term_lower, na=False) |
                filtered_df["Phone Number"].astype(str).str.contains(search_term_lower, na=False) |
                filtered_df["Skills"].str.lower().str.contains(search_term_lower, na=False) |
                filtered_df["Education"].str.lower().str.contains(search_term_lower, na=False) |
                filtered_df["Last Company"].str.lower().str.contains(search_term_lower, na=False) |
                filtered_df["Last Designation"].str.lower().str.contains(search_term_lower, na=False)
            )

            filtered_df = filtered_df[mask]
            st.info(f"Showing {len(filtered_df)} search results")

        # Show filtered or original resumes
        st.dataframe(filtered_df)

        csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered Resumes", csv_filtered, "filtered_resumes.csv", "text/csv")

    
    st.subheader("Job Description Match")
    job_desc = st.text_area("Enter a Brief Job Description", height = 200)

    if st.button("Find the Top 5 Candidates"):
        if not job_desc:
            st.warning("Please enter a job description")
        elif st.session_state.resumes_df.empty:
            st.warning("No Resumes Found")
        else: 
            with st.spinner("Matching Resumes..."):
                top_candidates = match_resumes_to_job(st.session_state.resumes_df, job_desc, client)
                st.session_state.top_candidates_df = top_candidates
                st.success("Found!")


    if "top_candidates_df" in st.session_state and not st.session_state.top_candidates_df.empty:
        st.subheader("Top 5 Matched Candidates")
        st.dataframe(st.session_state.top_candidates_df)
        csv_top = st.session_state.top_candidates_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Top 5", csv_top, "top_candidates.csv", "text/csv")