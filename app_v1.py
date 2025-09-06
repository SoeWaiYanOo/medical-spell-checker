# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re

# --- Core Logic Functions ---
def levenshtein_distance(word1, word2):
    len1, len2 = len(word1), len(word2)
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if word1[i-1] == word2[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + cost
            )
    return matrix[len1][len2]

# --- Data Loading Function ---
@st.cache_data
def load_models():
    """Loads the pre-processed dictionary and bigram models from CSV files."""
    word_dict_df = pd.read_csv("word_dictionary.csv", dtype={'word': str})
    bigram_counts = pd.read_csv("bigram_model.csv", dtype={'word1': str, 'word2': str})
    word_dict_df.dropna(subset=['word'], inplace=True)
    bigram_counts.dropna(subset=['word1', 'word2'], inplace=True)
    return word_dict_df, bigram_counts

# --- Spell Checker Class ---
class SpellChecker:
    def __init__(self, word_dict_df, bigram_model_df):
        self.word_dict = word_dict_df
        self.bigram_model = bigram_model_df
        self.vocab = set(word_dict_df['word'].tolist())

    def generate_candidates(self, word, max_distance=2):
        candidates = []
        for dict_word in self.word_dict['word']:
            distance = levenshtein_distance(word, dict_word)
            if 0 < distance <= max_distance:
                frequency = self.word_dict[self.word_dict['word'] == dict_word]['frequency'].iloc[0]
                candidates.append({
                    'word': dict_word,
                    'distance': distance,
                    'frequency': frequency
                })
        candidates.sort(key=lambda x: (x['distance'], -x['frequency']))
        return candidates[:10]

    def rank_candidates_by_context(self, candidates, previous_word):
        if not previous_word or not candidates:
            return [{'word': c['word'], 'distance': c['distance'], 'score': 0} for c in candidates[:5]]

        ranked_suggestions = []
        prev_word_count = self.word_dict[self.word_dict['word'] == previous_word]['frequency'].iloc[0] if previous_word in self.vocab else 0

        for candidate in candidates:
            score = 0
            if prev_word_count > 0:
                bigram_row = self.bigram_model[
                    (self.bigram_model['word1'] == previous_word) &
                    (self.bigram_model['word2'] == candidate['word'])
                ]
                if not bigram_row.empty:
                    bigram_count = bigram_row['frequency'].iloc[0]
                    score = bigram_count / prev_word_count
            ranked_suggestions.append({'word': candidate['word'], 'distance': candidate['distance'], 'score': score})

        ranked_suggestions.sort(key=lambda x: x['score'], reverse=True)
        return ranked_suggestions[:5]

    def check_non_word_errors(self, words):
        errors = []
        for i, word in enumerate(words):
            cleaned_word = re.sub(r'[^\w\s]', '', word).lower()
            if cleaned_word and cleaned_word not in self.vocab:
                errors.append({
                    'type': 'Non-Word',
                    'position': i,
                    'original_word': word,
                    'cleaned_word': cleaned_word
                })
        return errors

    def check_real_word_errors(self, words, threshold=0.00001):
        errors = []
        for i in range(len(words) - 1):
            prev_word = words[i].lower()
            current_word = words[i+1].lower()
            
            if prev_word in self.vocab and current_word in self.vocab:
                prev_word_count = self.word_dict[self.word_dict['word'] == prev_word]['frequency'].iloc[0]
                
                if prev_word_count > 0:
                    bigram_row = self.bigram_model[
                        (self.bigram_model['word1'] == prev_word) &
                        (self.bigram_model['word2'] == current_word)
                    ]
                    
                    prob = 0
                    if not bigram_row.empty:
                        prob = bigram_row['frequency'].iloc[0] / prev_word_count

                    if prob < threshold:
                        errors.append({
                            'type': 'Real-Word (Context)',
                            'position': i + 1,
                            'original_word': words[i+1],
                            'cleaned_word': current_word,
                            'context': f"'{prev_word} **{words[i+1]}**'"
                        })
        return errors


# --- UI HELPER FUNCTIONS ---
def highlight_text(words, errors):
    error_positions = {e['position']: e['type'] for e in errors}
    highlighted_words = []
    for i, word in enumerate(words):
        if i in error_positions:
            error_type = error_positions[i]
            color = "#FF4B4B" if error_type == "Non-Word" else "#FFA500" # Red for Non-Word, Orange for Real-Word
            highlighted_words.append(f'<span style="background-color: {color}; border-radius: 5px; padding: 2px 5px; font-weight: 600;">{word}</span>')
        else:
            highlighted_words.append(word)
    
    html_text = " ".join(highlighted_words)
    return re.sub(r'\s+([.,;?!])', r'\1', html_text)

def update_word(error_index, new_word):
    error_position = st.session_state.errors[error_index]['position']
    st.session_state.words[error_position] = new_word
    st.session_state.errors[error_index]['resolved'] = True

def ignore_error(error_index):
    st.session_state.errors[error_index]['resolved'] = True

# --- STREAMLIT PAGE CONFIG AND LAYOUT ---
st.set_page_config(page_title="Medical Spell Checker", layout="wide")

st.title("Advanced Medical Spell Checker")

# --- LOAD MODELS AND DISPLAY CORPUS INFO ---
with st.spinner("Loading dictionary and models..."):
    word_dictionary, bigram_model = load_models()
spell_checker = SpellChecker(word_dictionary, bigram_model)

st.success("Models loaded successfully!")
st.metric(label="**Total Unique Words in Medical Corpus**", value=f"{len(word_dictionary):,}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Text for Analysis")
    user_text = st.text_area(
        "Enter text to check:", 
        "The pateint has diabetis and needs treatmnt. The patience was discharged.", 
        height=250
    )
    char_count = len(user_text)
    st.caption(f"Characters: {char_count}")

with col2:
    st.subheader("Corpus Dictionary Explorer")
    search_term = st.text_input("Search for a word in the corpus:")
    if search_term:
        filtered_dict = word_dictionary[word_dictionary['word'].str.contains(search_term, case=False, na=False)]
    else:
        filtered_dict = word_dictionary.head(1000)
    
    st.dataframe(filtered_dict, height=180, use_container_width=True)

# --- INTERACTIVE UI BLOCK ---
if st.button("Check Spelling", key="main_check_button", use_container_width=True):
    if user_text:
        st.session_state.words = re.findall(r'\b\w+\b|[.,;?!]', user_text)
        non_word_errors = spell_checker.check_non_word_errors(st.session_state.words)
        real_word_errors = spell_checker.check_real_word_errors(st.session_state.words)
        errors = sorted(non_word_errors + real_word_errors, key=lambda x: x['position'])
        
        for error in errors:
            error['resolved'] = False
        st.session_state.errors = errors
    else:
        st.warning("Please enter some text to check.")

if 'errors' in st.session_state and st.session_state.errors:
    
    # --- CHANGE 1: New Subheader ---
    st.subheader("Result")
    highlighted_html = highlight_text(st.session_state.words, st.session_state.errors)
    st.markdown(highlighted_html, unsafe_allow_html=True)

    corrected_sentence = " ".join(st.session_state.words)
    corrected_sentence = re.sub(r'\s+([.,;?!])', r'\1', corrected_sentence)
    # --- CHANGE 2: New Label in the Green Box ---
    st.success(f"**Live Preview:** {corrected_sentence}")
    
    
    st.subheader("Interactive Corrections")
    unresolved_errors = [e for e in st.session_state.errors if not e.get('resolved', False)]
    if not unresolved_errors:
        st.success("✅ All errors have been resolved!")
    else:
        st.warning(f"Found {len(unresolved_errors)} unresolved spelling error(s).")

    for i, error in enumerate(st.session_state.errors):
        if not error.get('resolved', False):
            with st.container(border=True):
                candidates = spell_checker.generate_candidates(error['cleaned_word'])
                previous_word = st.session_state.words[error['position'] - 1].lower() if error['position'] > 0 else None
                final_suggestions = spell_checker.rank_candidates_by_context(candidates, previous_word)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.error(f"**{error['original_word']}**")
                    color = "red" if error['type'] == "Non-Word" else "orange"
                    st.markdown(f"Type: <span style='color:{color};'>{error['type']}</span>", unsafe_allow_html=True)

                with col2:
                    st.write("**Suggestions:**")
                    # --- CHANGE 2: SIMPLIFIED BUTTONS ---
                    top_suggestions = [s['word'] for s in final_suggestions[:4]] # Get top 4 suggestion words
                    
                    action_cols = st.columns(len(top_suggestions) + 1)
                    
                    for idx, suggestion in enumerate(top_suggestions):
                        with action_cols[idx]:
                            if st.button(suggestion, key=f"sugg_{i}_{suggestion}", use_container_width=True):
                                update_word(i, suggestion)

                    with action_cols[-1]:
                        if st.button("Ignore", key=f"ignore_{i}", use_container_width=True):
                            ignore_error(i)
