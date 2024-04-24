import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

APP_TITLE = 'Player Ability Prediction Model'
APP_SUB_TITLE = ':one: 선수의 전반적인 능력과 잠재력이 궁금하신가요?'

def predict_ability(scaler, model, columns, test_input):
    test_input = np.array(test_input).reshape(-1, 1)
    scaler = scaler.fit(test_input)
    test_input_scaled = scaler.transform(test_input).reshape(1,-1)
    input_df = pd.DataFrame(test_input_scaled, columns=columns)
    model_result = round(model.predict(input_df)[0], 2)
    return model_result

def main():
    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.subheader(APP_SUB_TITLE)
    
    # Load Scaler
    with open('data/OP_ST_Scaler.pkl', 'rb') as f:
        fw_scaler = pickle.load(f)
    with open('data/OP_MID_Scaler.pkl', 'rb') as f:
        mid_scaler = pickle.load(f)
    with open('data/OP_DC_Scaler.pkl', 'rb') as f:
        df_scaler = pickle.load(f)

    # Load Potential Model
    fw_potential_model = joblib.load('data/Potential_ST_Model.pkl')
    mid_potential_model = joblib.load('data/Potential_MID_Model.pkl')
    df_potential_model = joblib.load('data/Potential_DC_Model.pkl')

    # Load Overall Model
    fw_overall_model = joblib.load('data/Overall_ST_Model.pkl')
    mid_overall_model = joblib.load('data/Overall_MID_Model.pkl')
    df_overall_model = joblib.load('data/Overall_DC_Model.pkl')
    
    # Load Columns
    fw_columns = list(fw_overall_model.feature_names_in_)
    mid_columns = list(mid_overall_model.feature_names_in_)[:-1]
    df_columns = list(df_overall_model.feature_names_in_)[:-1]

    # Overall | Potential Prediction
    tab1, tab2, tab3 = st.tabs(["Forward", "Midfieldeer", "Defender"])

    with tab1:
        st.subheader("선수의 능력치를 입력하세요")
        st.markdown("")
        col_1, col_2, col_3, col_4, col_5 = st.columns(5)
        with col_1:
            fw_age = st.slider("Age", 0, 100, 25)
        with col_2:
            fw_dribbling = st.slider("Dribbling", 0, 100, 25)
        with col_3:
            fw_firsttouch = st.slider("First-touch", 0, 100, 25)
        with col_4:
            fw_finishing = st.slider("Finishing", 0, 100, 25)
        with col_5:
            fw_positioning = st.slider("Positioning", 0, 100, 25)
        
        col_6, col_7, col_8, col_9, col_10 = st.columns(5)
        with col_6:
            fw_vision = st.slider("Vision", 0, 100, 25)
        with col_7:
            fw_stamina = st.slider("Stamina", 0, 100, 25)
        with col_8:
            fw_technique = st.slider("Technique", 0, 100, 25)
        with col_9:
            fw_composure = st.slider("Composure", 0, 100, 25)
        with col_10:
            fw_balance = st.slider("Balance", 0, 100, 25)

        fw_input = [fw_age, fw_dribbling, fw_firsttouch, fw_finishing, fw_positioning,
                    fw_vision, fw_stamina, fw_technique, fw_composure, fw_balance]
        st.markdown("---")
        st.subheader("선수 능력 및 잠재력 예측하기")
        if not st.button("Click!"):
            st.error("선수의 능력치를 모두 입력하고 Click 버튼을 눌러주세요")
        else:
            fw_overall = predict_ability(fw_scaler, fw_overall_model, fw_columns, fw_input)
            fw_potential = predict_ability(fw_scaler, fw_potential_model, fw_columns, fw_input)

            col_11, col_12 = st.columns(2)
            col_11.metric("Overall Value", fw_overall)
            col_12.metric("Potential Value", fw_potential)
        
    with tab2:
        st.subheader("선수의 능력치를 입력하세요")
        st.markdown("")
        col_13, col_14, col_15, col_16, col_17 = st.columns(5)
        with col_13:
            mid_age = st.slider("Age", 0, 100, 50)
        with col_14:
            mid_composure = st.slider("Composure", 0, 100, 50)
        with col_15:
            mid_passing = st.slider("Passing", 0, 100, 50)
        with col_16:
            mid_anticipation = st.slider("Anticipation", 0, 100, 50)
        with col_17:
            mid_technique = st.slider("Technique", 0, 100, 50)
        
        col_18, col_19, col_20, col_21, col_22 = st.columns(5)
        with col_18:
            mid_firsttouch = st.slider("First-touch", 0, 100, 50)
        with col_19:
            mid_vision = st.slider("Vision", 0, 100, 50)
        with col_20:
            mid_concen = st.slider("Concentration", 0, 100, 50)
        with col_21:
            mid_stamina = st.slider("Stamina", 0, 100, 50)
        with col_22:
            mid_teamwork = st.slider("Teamwork", 0, 100, 50)

        mid_input = [mid_age, mid_composure, mid_passing, mid_anticipation, mid_technique,
                    mid_firsttouch, mid_vision, mid_concen, mid_stamina, mid_teamwork]
        st.markdown("---")
        st.subheader("선수 능력 및 잠재력 예측하기")
        if not st.button("Click!!"):
            st.error("선수의 능력치를 모두 입력하고 Click 버튼을 눌러주세요")
        else:
            mid_overall = predict_ability(mid_scaler, mid_overall_model, mid_columns, mid_input)
            mid_potential = predict_ability(mid_scaler, mid_potential_model, mid_columns, mid_input)

            col_23, col_24 = st.columns(2)
            col_23.metric("Overall Value", mid_overall)
            col_24.metric("Potential Value", mid_potential)
        
    with tab3:  
        st.subheader("선수의 능력치를 입력하세요")
        st.markdown("")
        col_1, col_2, col_3, col_4, col_5 = st.columns(5)
        with col_1:
            df_age = st.slider("Age", 0, 100, 75)
        with col_2:
            df_passing = st.slider("Passing", 0, 100, 75)
        with col_3:
            df_bravery = st.slider("Bravery", 0, 100, 75)
        with col_4:
            df_anticipation = st.slider("Anticipation", 0, 100, 75)
        with col_5:
            df_teamwork = st.slider("Teamwork", 0, 100, 75)
        
        col_6, col_7, col_8, col_9, col_10 = st.columns(5)
        with col_6:
            df_stamina = st.slider("Stamina", 0, 100, 75)
        with col_7:
            df_technique = st.slider("Tenchnique", 0, 100, 75)
        with col_8:
            df_concen = st.slider("Concentration", 0, 100, 75)
        with col_9:
            df_wr = st.slider("Work-rate", 0, 100, 75)
        with col_10:
            df_composure = st.slider("Composure", 0, 100, 75)

        df_input = [df_age, df_passing, df_bravery, df_anticipation, df_teamwork,
                    df_stamina, df_technique, df_concen, df_wr, df_composure]
        st.markdown("---")
        st.subheader("선수 능력 및 잠재력 예측하기")
        if not st.button("Click!!!"):
            st.error("선수의 능력치를 모두 입력하고 Click 버튼을 눌러주세요")
        else:
            df_overall = predict_ability(df_scaler, df_overall_model, df_columns, df_input)
            df_potential = predict_ability(fw_scaler, df_potential_model, df_columns, df_input)

            col_11, col_12 = st.columns(2)
            col_11.metric("Overall Value", df_overall)
            col_12.metric("Potential Value", df_potential)
        

    #Load Data



if __name__ == "__main__":
    main()