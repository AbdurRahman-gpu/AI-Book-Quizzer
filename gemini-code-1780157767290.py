import streamlit as st
import PyPDF2
from difflib import SequenceMatcher
import json
import google.generativeai as genai
import re

# --- 1. API CONFIGURATION ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-3.5-flash')

# --- 2. CORE UTILITY FUNCTIONS ---

def calculate_match_percentage(user_answer, book_answer):
    """Cleans up extra spaces and punctuation for a much fairer text match score."""
    clean_user = re.sub(r'[^\w\s]', '', user_answer.lower().strip())
    clean_book = re.sub(r'[^\w\s]', '', book_answer.lower().strip())
    
    clean_user = " ".join(clean_user.split())
    clean_book = " ".join(clean_book.split())
    
    match_ratio = SequenceMatcher(None, clean_user, clean_book).ratio()
    return round(match_ratio * 100, 2)

def generate_questions_via_llm(text, num_questions, q_type):
    """Calls the real Gemini API to generate questions based on the selected PDF section."""
    
    prompt = f"""
    You are an academic examiner creating a strict test. Read the following textbook excerpt and generate exactly {num_questions} {q_type} questions.
    
    CRITICAL INSTRUCTIONS:
    1. Focus ONLY on the core scientific/academic theory, formulas, definitions, and main textbook examples.
    2. Completely IGNORE any textbook design elements, page layout descriptions, symbols, prefaces, acknowledgements, or introductory metadata (e.g., do NOT ask about icons, page numbers, or magnifying glasses).
    3. Avoid obscure historical trivia or side-box anecdotes unless they are central to a core numerical/theoretical formula problem.
    4. Ensure questions test a student's actual understanding of the chapter's scientific/academic concepts.

    If the type is MCQ, return a JSON array of objects with keys: 
    'question', 'options' (list of exactly 4 alternative strings), 'correct_answer' (must match one of the options exactly), and 'explanation' (quoting the exact context/lines from the text).
    
    If the type is NAT, return a JSON array of objects with keys: 
    'question', 'book_text' (the exact answer wording from the text), and 'explanation' (the surrounding context lines).
    
    Return ONLY valid JSON. Do not include markdown code block formatting like ```json ... ```. Do not add conversational text outside the JSON structure.
    
    Text: {text}
    """
    
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()
        
        questions = json.loads(raw_text)
        return questions
    except Exception as e:
        st.error(f"Failed to generate or parse the AI's response. Error details: {e}")
        return []

# --- 3. STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="AI Book Quizzer", layout="wide")

st.title("📚 AI Book Quizzer")
st.markdown("Upload a textbook PDF, select your parameters, and test your knowledge!")

# 1. File Upload
uploaded_file = st.file_uploader("Upload your Book (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Read the file structure to dynamically discover total page count ceiling
    reader = PyPDF2.PdfReader(uploaded_file)
    total_pages = len(reader.pages)
    
    st.success(f"Book mapped successfully! Found a total of {total_pages} pages.")
    
    # 3. Quiz Configuration Panel (Sidebar Adjustments)
    st.sidebar.header("Quiz Settings")
    
    st.sidebar.subheader("Target Page Range")
    start_page = st.sidebar.number_input("Start Page", min_value=1, max_value=total_pages, value=1)
    end_page = st.sidebar.number_input("End Page", min_value=start_page, max_value=total_pages, value=min(start_page + 14, total_pages))
    
    st.sidebar.markdown(f"Currently targeting **{end_page - start_page + 1}** pages.")
    
    quiz_type = st.sidebar.radio("Select Question Type", ["MCQ", "NAT (Subjective/Fill-in)"])
    num_questions = st.sidebar.slider("Number of Questions", min_value=1, max_value=20, value=5)
    
    start_quiz = st.sidebar.button("Generate Quiz")

    if "answers_submitted" not in st.session_state:
        st.session_state.answers_submitted = False

    # 4. The Quiz Panel execution logic
    if start_quiz or "quiz_data" in st.session_state:
        if start_quiz:
            with st.spinner(f"Extracting text from page {start_page} to {end_page} and querying Gemini..."):
                # Dynamically compile text content only from within the boundaries set by the user
                targeted_text = ""
                for page_num in range(start_page - 1, end_page):
                    targeted_text += reader.pages[page_num].extract_text() + "\n"
                
                st.session_state.quiz_data = generate_questions_via_llm(targeted_text, num_questions, quiz_type)
                st.session_state.answers_submitted = False
                
        questions = st.session_state.quiz_data
        
        if not questions:
            st.warning("No questions available. Please try hitting 'Generate Quiz' again.")
        else:
            st.divider()
            st.header(f"Your Quiz Panel (Pages {start_page} - {end_page})")
            
            # Display MCQ Format
            if quiz_type == "MCQ":
                user_choices = {}
                for i, q in enumerate(questions):
                    st.subheader(f"Q{i+1}: {q.get('question', 'Missing Question Value')}")
                    options = q.get('options', ["A", "B", "C", "D"])
                    user_choices[i] = st.radio(f"Select answer for Q{i+1}", options, key=f"mcq_{i}", index=None)
                
                st.write("")
                if st.button("Submit MCQ Answers") or st.session_state.answers_submitted:
                    st.session_state.answers_submitted = True
                    for i, q in enumerate(questions):
                        correct = q.get('correct_answer')
                        explanation = q.get('explanation', 'No context available.')
                        
                        if user_choices[i] == correct:
                            st.success(f"**Q{i+1}: Correct!** \n\n*Explanation:* {explanation}")
                        elif user_choices[i] is None:
                            st.warning(f"**Q{i+1}: You left this blank.**")
                        else:
                            st.error(f"**Q{i+1}: Incorrect.** You selected {user_choices[i]}. \n\n*Explanation:* {explanation}")

            # Display NAT Format
            elif quiz_type == "NAT (Subjective/Fill-in)":
                user_inputs = {}
                for i, q in enumerate(questions):
                    st.subheader(f"Q{i+1}: {q.get('question', 'Missing Question Value')}")
                    user_inputs[i] = st.text_area(f"Type your answer for Q{i+1}", key=f"nat_{i}")
                
                st.write("")
                if st.button("Submit NAT Answers") or st.session_state.answers_submitted:
                    st.session_state.answers_submitted = True
                    for i, q in enumerate(questions):
                        user_ans = user_inputs[i]
                        book_ans = q.get('book_text', '')
                        explanation = q.get('explanation', 'No context available.')
                        
                        if user_ans.strip() == "":
                            st.warning(f"**Q{i+1}: You left this blank.**")
                            continue
                            
                        match_pct = calculate_match_percentage(user_ans, book_ans)
                        
                        if match_pct >= 60.0:
                            st.success(f"**Q{i+1}: Correct Answer! ({match_pct}% Wording Match)** \n\n*Book's exact wording:* '{book_ans}' \n\n*Context:* {explanation}")
                        else:
                            st.error(f"**Q{i+1}: Incorrect/Needs Improvement ({match_pct}% Wording Match)** \n\n*Book's exact wording:* '{book_ans}' \n\n*Context:* {explanation}")