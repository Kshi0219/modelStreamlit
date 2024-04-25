import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts
from unidecode import unidecode as ucd
import plotly.express as px

st.set_page_config(layout='wide')
st.title('경기 결과 예측!')
model=tf.keras.models.load_model('./data/match_pred_DL.h5')
gkPlayer=pd.read_csv('./data/GK.csv',encoding='utf-16')[['player_nm','player_overall','player_team','player_position']]
gkName=[]
for idx,rows in gkPlayer.iterrows():
    gkName.append(ucd(rows['player_nm']))
gkPlayer['player_nm']=gkName

ngkPlayer=pd.read_csv('./data/UNGK.csv',encoding='utf-16')[['player_nm','player_overall','player_team','player_position']]
ngkName=[]
for idx,rows in ngkPlayer.iterrows():
    ngkName.append(ucd(rows['player_nm']))
ngkPlayer['player_nm']=ngkName

with st.container(border=True):
    st.subheader('가상 드래프트 승부예측')
    st.divider()
    first_col1,first_col2,first_col3=st.columns(3)
    with first_col1:
        st.markdown('#### <center> 선수들의 포지션별 오버롤 평균 Input</center>',unsafe_allow_html=True)
        st.markdown('##### <center><b> 홈팀 GK, DC, MID, ST</b></center>',True)
        st.markdown('##### <center><b> 어웨이팀 GK, DC, MID, ST</b></center>',True)
    with first_col2:
        st.markdown('#### <center> 다중분류 딥러닝 알고리즘</center>',unsafe_allow_html=True)
        st.markdown('##### <center><b> Keras 기반</b></center>',True)
    with first_col3:
        st.markdown('#### <center> Output : 3개 클래스로 분류될 확률</center>',unsafe_allow_html=True)
        st.markdown('##### <center> 0 : 무승부</center>',unsafe_allow_html=True)
        st.markdown('##### <center> 1 : 어웨이팀 승리</center>',unsafe_allow_html=True)
        st.markdown('##### <center> 2 : 홈팀 승리</center>',unsafe_allow_html=True)
with st.container(border=True):
    st.subheader('넣고 싶은 선수를 선택하세요')
    draftcol1,draftcol2=st.columns(2)
    with draftcol1:
        st.markdown('##### **Home Team**')
        st.write('홈 팀 포메이션 : 4-3-3')
        st.divider()
    with draftcol2:
        st.markdown('##### **Away Team**')
        st.write('어웨이 팀 포메이션 : 4-3-3')
        st.divider()
    draftcol1_1,draftcol2_1=st.columns(2)
    with draftcol1_1:
            gk_433_player_home=st.selectbox('골키퍼',list(gkPlayer['player_nm']))
            gk_433_rating_home=gkPlayer.query(f"player_nm=='{gk_433_player_home}'")['player_overall'].iloc[0]
            st.write(gk_433_rating_home)
            col_433_df1,col_433_df2,col_433_df3,col_433_df4=st.columns(4)
            with col_433_df1:
                lb_433_player_home=st.selectbox('좌풀백',
                                           list(ngkPlayer.query("player_position=='DC'")['player_nm']))
                lb_433_rating_home=ngkPlayer.query(f"player_nm=='{lb_433_player_home}'")['player_overall'].iloc[0]
                st.write(lb_433_rating_home)
            with col_433_df2:
                lcb_433_player_home=st.selectbox('좌센터백',
                                           list(ngkPlayer.query("player_position=='DC'")['player_nm']))
                lcb_433_rating_home=ngkPlayer.query(f"player_nm=='{lcb_433_player_home}'")['player_overall'].iloc[0]
                st.write(lcb_433_rating_home)
            with col_433_df3:
                rcb_433_player_home=st.selectbox('우센터백',
                                           list(ngkPlayer.query("player_position=='DC'")['player_nm']))
                rcb_433_rating_home=ngkPlayer.query(f"player_nm=='{rcb_433_player_home}'")['player_overall'].iloc[0]
                st.write(rcb_433_rating_home)
            with col_433_df4:
                rb_433_player_home=st.selectbox('우풀백',
                                           list(ngkPlayer.query("player_position=='DC'")['player_nm']))
                rb_433_rating_home=ngkPlayer.query(f"player_nm=='{rb_433_player_home}'")['player_overall'].iloc[0]
                st.write(rb_433_rating_home)

            col_433_mf1,col_433_mf2,col_433_mf3=st.columns(3)
            with col_433_mf1:
                lcm_433_player_home=st.selectbox('좌측 미드필더',
                                           list(ngkPlayer.query("player_position=='MID'")['player_nm']))
                lcm_433_rating_home=ngkPlayer.query(f"player_nm=='{lcm_433_player_home}'")['player_overall'].iloc[0]
                st.write(lcm_433_rating_home)
            with col_433_mf2:
                cm_433_player_home=st.selectbox('중앙 미드필더',
                                           list(ngkPlayer.query("player_position=='MID'")['player_nm']))
                cm_433_rating_home=ngkPlayer.query(f"player_nm=='{cm_433_player_home}'")['player_overall'].iloc[0]
                st.write(cm_433_rating_home)
            with col_433_mf3:
                rcm_433_player_home=st.selectbox('우측 미드필더',
                                           list(ngkPlayer.query("player_position=='MID'")['player_nm']))
                rcm_433_rating_home=ngkPlayer.query(f"player_nm=='{rcm_433_player_home}'")['player_overall'].iloc[0]
                st.write(rcm_433_rating_home)
            
            col_433_fw1,col_433_fw2,col_433_fw3=st.columns(3)
            with col_433_fw1:
                lwf_433_player_home=st.selectbox('좌측 윙어',
                                           list(ngkPlayer.query("player_position=='ST'")['player_nm']))
                lwf_433_rating_home=ngkPlayer.query(f"player_nm=='{lwf_433_player_home}'")['player_overall'].iloc[0]
                st.write(lwf_433_rating_home)
            with col_433_fw2:
                cf_433_player_home=st.selectbox('중앙 공격수',
                                           list(ngkPlayer.query("player_position=='ST'")['player_nm']))
                cf_433_rating_home=ngkPlayer.query(f"player_nm=='{cf_433_player_home}'")['player_overall'].iloc[0]
                st.write(cf_433_rating_home)
            with col_433_fw3:
                rwf_433_player_home=st.selectbox('우측 윙어',
                                           list(ngkPlayer.query("player_position=='ST'")['player_nm']))
                rwf_433_rating_home=ngkPlayer.query(f"player_nm=='{rwf_433_player_home}'")['player_overall'].iloc[0]
                st.write(rwf_433_rating_home)
    with draftcol2_1:
            gk_433_player_away=st.selectbox('골키퍼',list(gkPlayer['player_nm']),index=1)
            gk_433_rating_away=gkPlayer.query(f"player_nm=='{gk_433_player_away}'")['player_overall'].iloc[0]
            st.write(gk_433_rating_away)
            col_433_df1_away,col_433_df2_away,col_433_df3_away,col_433_df4_away=st.columns(4)
            with col_433_df1_away:
                lb_433_player_away=st.selectbox('좌풀백',
                                           list(ngkPlayer.query("player_position=='DC'")['player_nm']),index=1)
                lb_433_rating_away=ngkPlayer.query(f"player_nm=='{lb_433_player_away}'")['player_overall'].iloc[0]
                st.write(lb_433_rating_away)
            with col_433_df2_away:
                lcb_433_player_away=st.selectbox('좌센터백',
                                           list(ngkPlayer.query("player_position=='DC'")['player_nm']),index=1)
                lcb_433_rating_away=ngkPlayer.query(f"player_nm=='{lcb_433_player_away}'")['player_overall'].iloc[0]
                st.write(lcb_433_rating_away)
            with col_433_df3_away:
                rcb_433_player_away=st.selectbox('우센터백',
                                           list(ngkPlayer.query("player_position=='DC'")['player_nm']),index=1)
                rcb_433_rating_away=ngkPlayer.query(f"player_nm=='{rcb_433_player_away}'")['player_overall'].iloc[0]
                st.write(rcb_433_rating_away)
            with col_433_df4_away:
                rb_433_player_away=st.selectbox('우풀백',
                                           list(ngkPlayer.query("player_position=='DC'")['player_nm']),index=1)
                rb_433_rating_away=ngkPlayer.query(f"player_nm=='{rb_433_player_away}'")['player_overall'].iloc[0]
                st.write(rb_433_rating_away)

            col_433_mf1_away,col_433_mf2_away,col_433_mf3_away=st.columns(3)
            with col_433_mf1_away:
                lcm_433_player_away=st.selectbox('좌측 미드필더',
                                           list(ngkPlayer.query("player_position=='MID'")['player_nm']),index=1)
                lcm_433_rating_away=ngkPlayer.query(f"player_nm=='{lcm_433_player_away}'")['player_overall'].iloc[0]
                st.write(lcm_433_rating_away)
            with col_433_mf2_away:
                cm_433_player_away=st.selectbox('중앙 미드필더',
                                           list(ngkPlayer.query("player_position=='MID'")['player_nm']),index=1)
                cm_433_rating_away=ngkPlayer.query(f"player_nm=='{cm_433_player_away}'")['player_overall'].iloc[0]
                st.write(cm_433_rating_away)
            with col_433_mf3_away:
                rcm_433_player_away=st.selectbox('우측 미드필더',
                                           list(ngkPlayer.query("player_position=='MID'")['player_nm']),index=1)
                rcm_433_rating_away=ngkPlayer.query(f"player_nm=='{rcm_433_player_away}'")['player_overall'].iloc[0]
                st.write(rcm_433_rating_away)
            
            col_433_fw1_away,col_433_fw2_away,col_433_fw3_away=st.columns(3)
            with col_433_fw1_away:
                lwf_433_player_away=st.selectbox('좌측 윙어',
                                           list(ngkPlayer.query("player_position=='ST'")['player_nm']),index=1)
                lwf_433_rating_away=ngkPlayer.query(f"player_nm=='{lwf_433_player_away}'")['player_overall'].iloc[0]
                st.write(lwf_433_rating_away)
            with col_433_fw2_away:
                cf_433_player_away=st.selectbox('중앙 공격수',
                                           list(ngkPlayer.query("player_position=='ST'")['player_nm']),index=1)
                cf_433_rating_away=ngkPlayer.query(f"player_nm=='{cf_433_player_away}'")['player_overall'].iloc[0]
                st.write(cf_433_rating_away)
            with col_433_fw3_away:
                rwf_433_player_away=st.selectbox('우측 윙어',
                                           list(ngkPlayer.query("player_position=='ST'")['player_nm']),index=1)
                rwf_433_rating_away=ngkPlayer.query(f"player_nm=='{rwf_433_player_away}'")['player_overall'].iloc[0]
                st.write(rwf_433_rating_away)
with st.container(border=True):
    st.subheader('어떤 팀이 이길까용?')
    df_mean_home=np.mean([lb_433_rating_home,lcb_433_rating_home,rcb_433_rating_home,rb_433_rating_home])
    mf_mean_home=np.mean([lcm_433_rating_home,cm_433_rating_home,rcm_433_rating_home])
    fw_mean_home=np.mean([lwf_433_rating_home,cf_433_rating_home,rwf_433_rating_home])
    df_mean_away=np.mean([lb_433_rating_away,lcb_433_rating_away,rcb_433_rating_away,rb_433_rating_away])
    mf_mean_away=np.mean([lcm_433_rating_away,cm_433_rating_away,rcm_433_rating_away])
    fw_mean_away=np.mean([lwf_433_rating_away,cf_433_rating_away,rwf_433_rating_away])
    inputDf=pd.DataFrame([[gk_433_rating_home,df_mean_home,mf_mean_home,fw_mean_home,gk_433_rating_away,df_mean_away,mf_mean_away,fw_mean_away]],
                         columns=['home_gk_mean','home_dc_mean','home_mid_mean','home_st_min','away_gk_mean','away_dc_mean','away_mid_mean','away_st_min'])
    prediction=pd.DataFrame(model.predict(inputDf),columns=['Draw(0)','Away(1)','Home(2)'])
    prediction['Draw(0)']=[round(i,2) for i in prediction['Draw(0)']]
    prediction['Away(1)']=[round(i,2) for i in prediction['Away(1)']]
    prediction['Home(2)']=[round(i,2) for i in prediction['Home(2)']]
    winnerIndex=prediction.iloc[0].tolist().index(prediction.iloc[0].max())
    st.markdown(f"### <center> {prediction.columns[winnerIndex].split('(')[0]} Win!!</center>",unsafe_allow_html=True)
    st.divider()
    final_col1,final_col2=st.columns(2)
    with final_col1:
        st.markdown('#### <center> Input 데이터프레임</center>',unsafe_allow_html=True)
        st.dataframe(inputDf.iloc[:,:4],hide_index=True,use_container_width=True)
        st.dataframe(inputDf.iloc[:,4:],hide_index=True,use_container_width=True)
    with final_col2:
        st.markdown('#### <center> Output : 3개 클래스로 분류될 확률</center>',unsafe_allow_html=True)
        st.markdown(f"##### <center> 0 : 무승부 -> {prediction['Draw(0)'][0]*100} %</center>",unsafe_allow_html=True)
        st.markdown(f"##### <center> 1 : 어웨이팀 승리 -> {prediction['Away(1)'][0]*100} %</center>",unsafe_allow_html=True)
        st.markdown(f"##### <center> 2 : 홈팀 승리 -> {prediction['Home(2)'][0]*100} %</center>",unsafe_allow_html=True)