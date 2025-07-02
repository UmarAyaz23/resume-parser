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
import datetime
import plotly.express as px
from collections import Counter
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
        - Education (Degree + Institute + Start and Completion Years)
        - Total Experience (number of years)
        - Last Company
        - Last Designation

        Only output a JSON like this:
        {{
        "Full Name": "John Doe",
        "Email": "john@example.com",
        "Phone Number": "123-456-7890",
        "Skills": "Python, Java, SQL",
        "Education": "BS Computer Science - XYZ University, From Year to Year",
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


def extract_edu_details(education_str):
    start_year = None
    end_year = None
    university = ""

    # Extract years
    years = re.findall(r'(20\d{2})', education_str)
    if len(years) >= 1:
        start_year = int(years[0])
    if len(years) >= 2:
        end_year = int(years[1])
    elif start_year:
        end_year = start_year + 4  # Assume 4 years if end year missing

    # Extract university (everything after the dash, before comma if present)
    uni_match = re.search(r'-(.*?)(?:,|$)', education_str)
    if uni_match:
        university = uni_match.group(1).strip()

    return pd.Series({
        "Start Year": start_year,
        "End Year": end_year,
        "University": university
    })

def compute_semesters_completed(start_year, end_year):
    current_year = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    if pd.isna(start_year) or pd.isna(end_year):
        return None

    # Approximate semester based on year difference
    years_completed = current_year - start_year

    # Add 1 if current month is June or later (2nd semester)
    semesters = years_completed * 2

    # Cap at 8
    return min(semesters, 8)


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
st.set_page_config(layout = "wide")
st.markdown("""
    <style>
    h1 {
        text-align: center;
    }
    h3 {
        text-align: center;
    }
    .stDataFrame {
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

left, center, right = st.columns([1, 6, 1])

with center:
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


        #---------------------------------GRAPH VISUALIZATIONS---------------------------------
        st.subheader("Graph Insights")
        df = st.session_state.resumes_df.copy()


        #---------------PIE CHART: SKILS---------------
        skill_counter = Counter()
        for skills in df['Skills'].dropna():
            for skill in map(str.strip, skills.split(',')):
                skill_counter[skill.lower()] += 1
        
        top_skills_df = pd.DataFrame(skill_counter.items(), columns = ['Skill', 'Count']).sort_values(by = 'Count', ascending = False).head(10)
        st.plotly_chart(px.pie(top_skills_df, names = "Skill", values = "Count", title = "Top 10 Skills", height = 500))


        #---------------BAR CHART: EDUCATION---------------
        edu_details = df["Education"].apply(extract_edu_details)
        df = pd.concat([df, edu_details], axis=1)

        # Calculate number of semesters completed
        df["Semesters Completed"] = df.apply(
            lambda row: compute_semesters_completed(row["Start Year"], row["End Year"]), axis=1
        )

        # Drop missing
        df_edu_graph = df.dropna(subset=["Semesters Completed", "Full Name", "University"])

        # Plot bar chart
        fig = px.bar(
            df_edu_graph,
            x="Full Name",
            y="Semesters Completed",
            hover_data=["University"],
            color="Semesters Completed",
            color_continuous_scale = "reds",
            title="Education Progress by Semester",
            labels={"Semesters Completed": "Semesters Completed"},
            height = 500
        )
        fig.update_layout(yaxis=dict(tickmode="linear", dtick=1, range=[0, 8]))
        st.plotly_chart(fig)


        #---------------BAR CHART: EXPERIENCE---------------
        df['Total Experience'] = pd.to_numeric(df['Total Experience'], errors = "coerce")
        df_exp = df[['Full Name', 'Total Experience']].dropna()
        st.plotly_chart(px.bar(df_exp, x = "Full Name", y = "Total Experience", color_continuous_scale = "blues", title = "Experience (Years) Per Candidate", labels = {'Total Experience': 'Years'}))


        # ------------------ INTERACTIVE SKILLSET CHART ------------------
        df_skills = df[['Full Name', 'Skills']].dropna()
        skill_set = set()

        for skills in df_skills['Skills']:
            for skill in map(str.strip, skills.lower().split(',')):
                skill_set.add(skill)

        skill_options = sorted(skill_set)

        # Step 2: Multiselect UI for skills
        selected_skills = st.multiselect("Select Skills", options=skill_options)

        # Step 3: Filter candidates who have any selected skill
        if selected_skills:
            selected_skills_set = set(selected_skills)
            skill_data = []

            for _, row in df_skills.iterrows():
                candidate = row['Full Name']
                candidate_skills = set(map(str.strip, row['Skills'].lower().split(',')))

                for skill in selected_skills:
                    if skill in candidate_skills:
                        skill_data.append({
                            "Candidate": candidate,
                            "Skill": skill.capitalize(),  # Format for display
                            "Has Skill": 1
                        })

            skill_df = pd.DataFrame(skill_data)

            if not skill_df.empty:
                # Step 4: Create grouped bar chart
                fig = px.bar(
                    skill_df,
                    x="Candidate",
                    y="Has Skill",
                    color="Skill",
                    barmode="group",
                    title="Candidates with Selected Skill(s)",
                    labels={"Has Skill": "Skill Presence"},
                    height=500
                )

                # Clean up Y axis
                fig.update_yaxes(showticklabels=False, title=None)
                st.plotly_chart(fig)
            else:
                st.warning("No candidates found with the selected skill(s).")
        else:
            st.info("Select at least one skill to display the chart.")



        #---------------------------------JOB DESCRIPTION MATCH---------------------------------
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

    st.markdown("</div>", unsafe_allow_html=True)