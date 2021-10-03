import streamlit as st
import pandas as pd
import os

# set current SCRIPT file directory as a working directory
os.chdir( os.path.dirname( os.path.abspath(__file__) ) )
st.set_page_config(layout="wide")
pd.set_option('display.max_rows', None)

def main():
    st.title('[Demo] 의미·구조적 유사성을 가진 한국어 문자열 생성 기술 연구')
    st.subheader('Controllable Text Generator')

    # Applying styles to the buttons
    st.markdown("""<style>
                    .st-eb {
                        background-color:#f63366
                    } </style>""", unsafe_allow_html=True)

    # Select Box for the model
    st.sidebar.image('./fig/logo.png', width=150)
    model_name = st.sidebar.selectbox("Model", ("t5", "bart", "gpt2"))
    num_return_sequences = st.sidebar.slider("Number of return sentences", 0, 100)

    st.sidebar.text('')
    st.sidebar.text('')

    st.sidebar.markdown('### 📃 Template')
    st.sidebar.markdown('1. weather.general(day.p=\*,location=\*)')
    st.sidebar.markdown('2. weather.humidity(day.p=\*,location=서울,ti_range.p=점심)')
    st.sidebar.markdown('3. weather.humidity(day.p=\*,location=\*)')
    st.sidebar.markdown('4. weather.rainfall(day.p=내일,location=\*)')
    st.sidebar.markdown('5. weather.sunset(day.p=\*,location=논산)')
    st.sidebar.markdown('6. weather.temperature(day.p=내일,location=울산)')
    st.sidebar.markdown('7. weather.temperature(day.p=\*,location=\*,time=5시)')
    st.sidebar.markdown('8. weather.uv(day.p=\*,ti_range.p=\*)')
    st.sidebar.markdown('9. weather.uv(day.p=\*,location=\*)')
    st.sidebar.markdown('10. weather.windchill(location=*)')

    # vocab 
    intent_label_vocab = load_vocab(os.path.join('./data', "intent.label.vocab")) 
    slot_tag_label_vocab = load_vocab(os.path.join('./data', "slot_tag.label.vocab"))
    slot_value_label_vocab  = load_slot_value_vocab(os.path.join('./data', "slot_value.label.vocab"))

    row3_spacer1, row3_1, row3_spacer2 = st.beta_columns((.2, 7.1, .2))
    with row3_1:
        st.markdown("") 
        see_image = st.beta_expander('You can click here to see the overall architecture 👉')
        with see_image:
            st.image('./fig/overall_1.png', width=500)
            st.image('./fig/overall_2.png', width=500)
        see_data = st.beta_expander('You can click here to see the slot tags and slot values lists 👉')
        with see_data:
            df = pd.DataFrame( columns = ['Slot tag', 'Slot value'])
            for key, value in slot_value_label_vocab.items():
                value_str = ', '.join(value)
                df=df.append({'Slot tag' : key, 'Slot value' : value_str} , ignore_index=True)
            st.table(df)
    st.text('')

    row13_spacer1, row13_1, row13_spacer2, row13_2, row13_spacer3 = st.beta_columns((.2, 2.3, .2, 2.3, .2))
    with row13_1:
        intent_option = st.selectbox('Select a intent', intent_label_vocab)
        st.write('Selected intent : ', intent_option)
    with row13_2:
        slot_option = st.text_area("Enter the slot tags and values", "day.p=*,location=*")
        _slot_option = ''
        if '*' in slot_option:
            _slot_option = slot_option.replace('*', '\*')
        st.write('slot tags and values : ', _slot_option)

    semantic_control_grammar = intent_option + '(' + slot_option + ')'
    st.info('semantic control grammar : '+ intent_option + '(' + _slot_option + ')')

    # Generate button
    if st.button("Generate"):
        # Checking for exceptions
        if not check_exceptions(num_return_sequences):
            # Calling the forward method on click of Generate
            with st.spinner('Progress your text .... '):
                df = pd.read_csv(os.path.join('./data', model_name, "reranking_100.csv"), delimiter='\t')
                is_exist = df['query'] == semantic_control_grammar
                filtered = df[is_exist]
                filtered.rename(columns = {'generated_texts' : '생성 문장'}, inplace = True)
                filtered.reset_index(inplace = True) 
                filtered = filtered[:num_return_sequences]    

                html_table = filtered[['생성 문장', 'train_texts']].to_html(col_space='100px', justify='center') 
                st.table(data=filtered[['생성 문장', 'train_texts']])



def check_exceptions(num_return_sequences):
    # Checking for zero on the num of return sequences
    if num_return_sequences == 0:
        st.error("Please set the number of return sequences to more than one")
        return True
    return False

def load_vocab(fn):
    vocab = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            symbol, _id = line.split('\t')
            vocab.append(symbol)

    #vocab.sort()
    return vocab[1:]

def load_slot_value_vocab(fn):
    vocab = {}
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            slot_tag, slot_value = line.split('\t')
            slot_value = slot_value.replace('{', '').replace('}', '').replace("'", '')
            slot_value = slot_value.split(',')
            vocab[slot_tag] =  slot_value

    for key, value in vocab.items():
        for idx, v in enumerate(value):
            value[idx] = v.lstrip().rstrip()
        value.sort()

    return vocab 

if __name__ == '__main__':
    main()