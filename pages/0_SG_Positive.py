from typing import Any
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from bertopic import BERTopic
from gensim.models import LdaMulticore
from openai import OpenAI
import json


st.set_page_config(
        page_title="BERT_SG_Positive",
        page_icon=":cityscape:",
        layout="wide", 
        # initial_sidebar_state="expanded",
        initial_sidebar_state="collapsed",        
    )

testBModel = BERTopic.load("./models/sgPosBERT")
r = testBModel.visualize_barchart()
p = testBModel.get_topic_info()
lda = LdaMulticore.load("./models/sgPosLDAA/sgPosLDA")
topics = lda.print_topics(-1)
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
topicForOenAI = json.dumps(lda.print_topics())

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# # Set a default model
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Add user message to chat history
# st.session_state.messages.append({"role": "user", "content": topics})

# # Display assistant response in chat message container
# with st.chat_message("assistant"):
#     message_placeholder = st.empty()
#     full_response = ""

# for response in client.chat.completions.create(
#         model=st.session_state["openai_model"],
#         messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
#         stream=True,
#     ):
#         full_response += (response.choices[0].delta.content or "")
#         message_placeholder.markdown(full_response + "▌")
#     message_placeholder.markdown(full_response)
# st.session_state.messages.append({"role": "assistant", "content": full_response})


response = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with a list of topics broken down into tokens for each topic . These topics is the result of topic modelling on Airbnb listings' reviews. Your job is to summarise these output from LDA's topic model, and similar topics can integrate them together into one. \n Then output them into comprehensible points."
    },
    {
      "role": "user",
      "content": topicForOenAI
    }
  ],
  temperature=0.7,
  max_tokens=450,
  top_p=1
)

def generate_response(t):
    # llm = OpenAI(temperature=0.7, openai_api_key=st.secrets["OPENAI_API_KEY"])
    # st.info(llm(input_text))
    st.info(t)


def sgPosBERT() -> None:

    st.write("### using BERT	:arrow_heading_down:")

    st.plotly_chart(r, use_container_width=False, sharing="streamlit", theme="streamlit")
    st.write("")
    st.write("")
    st.subheader("Topic Information :arrow_heading_down:")
    st.write("")
    st.dataframe(data=p, 
                #  width=None, height=None, 
                 use_container_width=True, hide_index=True, column_order=("Topic", "Count", "Name", "Representation"), 
                 column_config=None
                )

def sgPosLDA() -> None:

    st.write("### using LDA :arrow_heading_down:")
    st.write("")
    string = """
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css">

    <div id="ldavis_el1849622229530641444586857286" style="margin-left:auto; width: 1210px; height: 780px; "></div>
    <script type="text/javascript" style="padding:100px">

    var ldavis_el1849622229530641444586857286_data = {"mdsDat": {"x": [0.15835960985412725, -0.15835960985412725], "y": [0.0, 0.0], "topics": [1, 2], "cluster": [1, 1], "Freq": [54.56756640920345, 45.432433590796556]}, "tinfo": {"Term": ["host", "time", "walk", "house", "recommend", "experience", "make", "small", "come", "help", "location", "family", "food", "nice", "home", "restaurant", "feel", "bathroom", "bed", "give", "helpful", "friendly", "price", "next", "kind", "value", "wonderful", "bit", "space", "book", "walk", "small", "restaurant", "value", "price", "shower", "distance", "money", "bit", "chinatown", "min", "street", "air", "toilet", "road", "kitchen", "window", "noise", "shop", "bar", "access", "building", "store", "door", "machine", "option", "mall", "cheap", "noisy", "washing", "bathroom", "sleep", "bed", "central", "hostel", "train", "transport", "food", "space", "public", "water", "floor", "little", "location", "stop", "good", "station", "area", "room", "hotel", "minute", "great", "bus", "close", "easy", "clean", "night", "staff", "check", "get", "apartment", "convenient", "need", "nice", "comfortable", "host", "house", "home", "help", "experience", "kind", "family", "beautiful", "welcome", "friend", "landlord", "hospitality", "meet", "warm", "wonderful", "first", "response", "accommodate", "respond", "care", "question", "environment", "tip", "welcoming", "owner", "reply", "hospitable", "hope", "answer", "information", "time", "give", "next", "trip", "book", "come", "arrive", "feel", "make", "visit", "recommend", "lovely", "helpful", "amazing", "friendly", "nice", "love", "airport", "go", "clean", "great", "comfortable", "apartment", "enjoy", "room", "good", "need", "convenient", "check", "day", "location"], "Freq": [7206.0, 3963.0, 4488.0, 2478.0, 5889.0, 2049.0, 2832.0, 2616.0, 2424.0, 1890.0, 10448.0, 1811.0, 2886.0, 7835.0, 1688.0, 2164.0, 2054.0, 2184.0, 2213.0, 1573.0, 3798.0, 3860.0, 1728.0, 1409.0, 1216.0, 1550.0, 1177.0, 1460.0, 1720.0, 1177.0, 4487.104273875886, 2615.9671094351233, 2163.225322362988, 1550.284801263078, 1728.0909432247563, 1124.1716846284705, 1356.7604258652068, 1040.7062173235963, 1459.194401699043, 873.129962065723, 1217.2596671201372, 810.7095129933892, 851.241533665161, 731.2970623364849, 699.2386994425077, 1180.3110580504149, 637.0560196004524, 581.0529292959407, 957.8522007582972, 586.6203882234823, 1039.0564808290312, 745.1849609701133, 693.4596771993323, 804.503470315591, 528.788693695675, 571.2227445968695, 782.2809783097118, 550.4222942988786, 427.61569208490636, 477.28789077864286, 2172.809517580253, 1011.2255522507768, 2190.6839291728, 772.6395503637651, 993.9808860929339, 637.6236496006804, 720.7641388413996, 2808.4300216770803, 1678.4387698054181, 951.8227389723382, 852.2915626366193, 856.0120608393337, 1582.121155559405, 8644.400684082684, 1442.8446304424745, 8097.399683192192, 3150.0223884911456, 2200.368003536447, 7572.37308340809, 1413.4673056745744, 1967.4699830617817, 7047.652548928988, 2126.1422225615047, 2016.005562014208, 2429.5221629882667, 5321.398739211068, 1757.6824889835962, 1769.8573613665362, 2270.077219844941, 1975.4212531259518, 2244.39468181633, 1959.2598677840929, 1956.0558238803667, 2305.4361682432927, 1485.1224347582402, 7204.135082973319, 2477.2375039717085, 1687.3783361681947, 1889.6106632014294, 2048.189450319896, 1215.2230357960066, 1809.6738072296255, 1027.630398935412, 848.1204930510085, 801.8340834114946, 698.9021228284364, 654.0752052367147, 831.4795460650745, 693.8761842724892, 1175.6999376239962, 1010.6609350172058, 601.7457101610387, 1035.2048137685915, 785.6208156586613, 490.8843676710518, 745.7932321031525, 495.13023289111, 400.96628836993665, 371.73587166204004, 446.10304312475523, 378.72604197915547, 329.06062676597946, 349.1349976613961, 426.6701138898608, 365.15187054079314, 3908.1465202795093, 1554.710087056641, 1391.5749800419885, 1050.3526142465662, 1151.9521360878755, 2310.6158154675236, 793.0414182878055, 1949.9202791713806, 2643.446076258431, 1250.1040826794178, 4736.782727595246, 1158.100507495753, 3050.118751084602, 1455.1816010010818, 3030.4470501445476, 5529.707333001433, 1482.1162843692234, 1158.8773388882314, 2034.8192640193042, 5313.058237999599, 5670.629981711681, 2526.4129483248234, 2712.9691924824165, 1476.575648702282, 3123.16848422627, 2728.8835324690262, 1790.3827001540892, 1785.2443742265752, 1831.280176494509, 1451.4171386477854, 1803.6005665738028], "Total": [7206.0, 3963.0, 4488.0, 2478.0, 5889.0, 2049.0, 2832.0, 2616.0, 2424.0, 1890.0, 10448.0, 1811.0, 2886.0, 7835.0, 1688.0, 2164.0, 2054.0, 2184.0, 2213.0, 1573.0, 3798.0, 3860.0, 1728.0, 1409.0, 1216.0, 1550.0, 1177.0, 1460.0, 1720.0, 1177.0, 4488.35548704011, 2616.7174230451305, 2164.0034296877457, 1550.9761339863564, 1728.8842480638111, 1124.8275818127383, 1357.5643602905434, 1041.401713184244, 1460.257500232511, 873.8384576108833, 1218.2677587506773, 811.4074871477069, 852.0168995195119, 732.0401262490368, 699.9690693616085, 1181.5746431846933, 637.7851461514377, 581.7297912687117, 958.9695478718061, 587.3055384794675, 1040.288864067103, 746.0779934600611, 694.3351909732509, 805.5248850108807, 529.4941709750369, 571.9891972174814, 783.3603610356811, 551.2265235605554, 428.244886956442, 477.9951642631035, 2184.992424924222, 1013.463335450041, 2213.560439120045, 774.1088376919038, 999.6537865381944, 638.7865937798814, 723.0997337327511, 2886.4892006308364, 1720.6687556442885, 960.8236925277298, 858.4491931217151, 862.4716286626234, 1645.8841488286398, 10448.001250656487, 1524.3913428850185, 10826.283215661218, 3783.148155408683, 2529.2062998817055, 10695.54156763436, 1562.4305015021112, 2417.577340892797, 12718.282530640668, 2774.8094030545967, 2654.8077660059967, 3481.3620573690014, 10634.456977210666, 2303.531382750677, 2612.7041902174938, 4101.35739633945, 3253.4065637305625, 4957.363874298746, 3744.5042420106683, 3746.438524034456, 7835.1435012447255, 4011.5353830830636, 7206.187311452254, 2478.037661005275, 1688.0176872473521, 1890.4510020620396, 2049.1733900190607, 1216.028947343646, 1811.0989078715734, 1028.470805034628, 848.8172179754187, 802.5546729105515, 699.534168076962, 654.7230499279436, 832.3145353521918, 694.59648749443, 1177.0040396786956, 1011.7930931088382, 602.4381715904774, 1036.4267376925711, 786.7300834111375, 491.58489006774465, 746.9435215616671, 495.92532910296984, 401.66917793977234, 372.4080028290417, 446.9220814710539, 379.4553451548769, 329.6979548322547, 349.8312456383529, 427.54626624101905, 365.92060789130534, 3963.8962700032216, 1573.4453343598043, 1409.0972709024486, 1065.8396772516687, 1177.4438371916287, 2424.991591740609, 809.5980517105205, 2054.6982164638566, 2832.5172849071805, 1316.14755353198, 5889.1223093082035, 1241.944284064308, 3798.4325993730454, 1622.9122641039105, 3860.7553115528226, 7835.1435012447255, 1714.570952067415, 1300.7428569463698, 2758.952840697423, 10634.456977210666, 12718.282530640668, 4011.5353830830636, 4957.363874298746, 1885.5243856958473, 10695.54156763436, 10826.283215661218, 3746.438524034456, 3744.5042420106683, 4101.35739633945, 2538.8192055480686, 10448.001250656487], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -4.0802, -4.6198, -4.8099, -5.143, -5.0344, -5.4644, -5.2764, -5.5416, -5.2036, -5.7171, -5.3848, -5.7913, -5.7425, -5.8944, -5.9392, -5.4157, -6.0323, -6.1244, -5.6245, -6.1148, -5.5431, -5.8756, -5.9475, -5.799, -6.2186, -6.1414, -5.827, -6.1785, -6.431, -6.3211, -4.8054, -5.5703, -4.7972, -5.8394, -5.5875, -6.0315, -5.9089, -4.5488, -5.0636, -5.6308, -5.7413, -5.7369, -5.1227, -3.4245, -5.2148, -3.4899, -4.434, -4.7928, -3.5569, -5.2354, -4.9047, -3.6288, -4.8271, -4.8803, -4.6938, -3.9097, -5.0175, -5.0106, -4.7616, -4.9007, -4.773, -4.9089, -4.9105, -4.7462, -5.186, -3.4236, -4.4911, -4.8751, -4.7619, -4.6813, -5.2033, -4.8051, -5.371, -5.563, -5.6191, -5.7565, -5.8228, -5.5828, -5.7637, -5.2364, -5.3876, -5.9062, -5.3636, -5.6395, -6.1098, -5.6915, -6.1012, -6.3121, -6.3878, -6.2054, -6.3692, -6.5098, -6.4505, -6.25, -6.4057, -4.0352, -4.9569, -5.0678, -5.3491, -5.2568, -4.5607, -5.6301, -4.7304, -4.4262, -5.175, -3.8429, -5.2515, -4.2831, -5.0231, -4.2895, -3.6881, -5.0048, -5.2508, -4.6878, -3.7281, -3.6629, -4.4714, -4.4002, -5.0085, -4.2594, -4.3943, -4.8158, -4.8187, -4.7932, -5.0257, -4.8085], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.6055, 0.6054, 0.6054, 0.6053, 0.6053, 0.6051, 0.6051, 0.6051, 0.605, 0.6049, 0.6049, 0.6049, 0.6048, 0.6047, 0.6047, 0.6047, 0.6046, 0.6046, 0.6046, 0.6046, 0.6045, 0.6045, 0.6045, 0.6045, 0.6044, 0.6044, 0.6044, 0.6043, 0.6043, 0.6042, 0.6001, 0.6035, 0.5953, 0.6038, 0.6, 0.6039, 0.6025, 0.5783, 0.5809, 0.5963, 0.5985, 0.5982, 0.5662, 0.4162, 0.5508, 0.3153, 0.4226, 0.4664, 0.2604, 0.5055, 0.3997, 0.0154, 0.3395, 0.3305, 0.246, -0.0866, 0.3353, 0.2162, 0.0142, 0.1068, -0.1867, -0.042, -0.0441, -0.6176, -0.3879, 0.7887, 0.7886, 0.7886, 0.7885, 0.7885, 0.7883, 0.7882, 0.7881, 0.7881, 0.788, 0.788, 0.788, 0.7879, 0.7879, 0.7878, 0.7878, 0.7878, 0.7878, 0.7875, 0.7875, 0.7874, 0.7873, 0.7872, 0.7871, 0.7871, 0.787, 0.787, 0.787, 0.7869, 0.7868, 0.7748, 0.777, 0.7764, 0.7743, 0.7671, 0.7406, 0.7683, 0.7366, 0.7199, 0.7375, 0.5712, 0.719, 0.5695, 0.6799, 0.5468, 0.4405, 0.6433, 0.6735, 0.4845, 0.095, -0.0188, 0.3266, 0.1861, 0.5445, -0.442, -0.5891, 0.0506, 0.0482, -0.0174, 0.2298, -0.9677]}, "token.table": {"Topic": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], "Freq": [0.9987610517505069, 0.0009612714646299393, 0.0009648535334261357, 0.9986234070960505, 0.99880647963663, 0.00117368563999604, 0.10916838731166273, 0.8910293020719514, 0.10351761072725706, 0.8965364500485655, 0.0023389281557572384, 0.9987223225083408, 0.4526599331620436, 0.547266666073362, 0.8698380990522193, 0.13008033390371826, 0.020998074246945584, 0.9794984045781088, 0.9994797623052244, 0.0017026912475387127, 0.9945114569792441, 0.005492009886677833, 0.0009723173425096209, 0.9995422280998904, 0.9898080762913285, 0.01039050011624854, 0.9991388503518656, 0.0006848107267661862, 0.021232435221393287, 0.9783906150018026, 0.9985551196128146, 0.0013403424424333083, 0.7661787500286085, 0.23388993827307944, 0.002034236650077246, 0.9988101951879277, 0.9985675945837152, 0.0012918080136917402, 0.9977749191881536, 0.0018141362167057337, 0.5534752962582641, 0.44643756275281127, 0.999040489001622, 0.0011443762760614228, 0.5003546501154454, 0.4996023785121897, 0.7593770162247772, 0.24069539353553204, 0.04701047227886393, 0.9529929950566187, 0.37018245090454716, 0.6296840882053105, 0.5231667193807438, 0.4766986187313056, 0.42815179498586764, 0.5715255331412088, 0.999584284688777, 0.0007366133269629897, 0.9993483937980717, 0.001241426576146673, 0.6980026667598153, 0.3021805783668007, 0.21691578380147003, 0.7833364613075091, 0.0020164325984494496, 0.9981341362324776, 0.00048800165221289465, 0.9994273837320082, 0.000552150959648699, 0.9993932369641453, 0.05110239506641778, 0.9490444798049016, 0.0009883443629046698, 0.9992161508966212, 0.9924964155949587, 0.0069567505766001785, 0.9728080740389804, 0.02702244650108279, 0.0012460210297865336, 0.9993088658887999, 0.21498383943585594, 0.7848205222778838, 0.6070560076990008, 0.3928190267540876, 0.01207541157299286, 0.9882771050528366, 0.262418403576983, 0.7375986896121, 0.7479021044163099, 0.2520717355751648, 0.5541628740374401, 0.4458935384032808, 0.0005289743023803495, 0.9997614314988605, 0.19692333098748732, 0.8029627800960378, 0.0005924108541959051, 0.9993971110284918, 0.0028585211082996736, 0.9976238667965861, 0.0030330791724467467, 0.9978830477349797, 0.0015273633639598548, 0.9988956400297451, 0.00027753927473152826, 0.9996964675829647, 0.9943442553668771, 0.006002078000202478, 0.9043602250734035, 0.09536424170979273, 0.0004035451178713427, 0.9995812569673158, 0.0027328332387801576, 0.9974841321547575, 0.0008223488447248313, 0.9991538463406701, 0.9986673349891386, 0.0008463282499907955, 0.001429522739037932, 0.9992363945875146, 0.9611855130422724, 0.038884875369598886, 0.8273352761569458, 0.17266460414011225, 0.1353108191412297, 0.8643561808935449, 0.06763588437728216, 0.9324089774868183, 0.9990667112083086, 0.0018885949172179747, 0.06672510032227161, 0.9330922759352585, 0.9982634288083169, 0.0012765516992433719, 0.0012014688648647144, 0.9984206267025776, 0.9989593759322849, 0.0008208376137487961, 0.813624435805474, 0.1861367545055736, 0.9996142572273904, 0.0009602442432539774, 0.5220958484842899, 0.4777871006067888, 0.012774135875284181, 0.9878665076886433, 0.2941873367901708, 0.705794348134336, 0.763176057927524, 0.23702737635291699, 0.9987454806687482, 0.001719011154335195, 0.9994281614004025, 0.0023351125266364544, 0.9982706015737823, 0.0017482847663288657, 0.002237526498374119, 0.9979368182748571, 0.9994885440914848, 0.0005784077222751648, 0.9908165331513459, 0.009366963023489615, 0.001338789307535939, 0.9987368234218105, 0.19561488783807, 0.8043643434799805, 0.002635356209284242, 0.9988000033187276, 0.0012710839728718112, 0.9990720026772436, 0.0016599213780891947, 0.9992726696096952, 0.9995363086425928, 0.00046210647648756023, 0.9986155540237052, 0.0014286345551125968, 0.7079585406795604, 0.2919908244244938, 0.9989889690721069, 0.0010427859802422828, 0.9992642589618894, 0.0008890251414251686, 0.9975693886854357, 0.0019734310359751446, 0.9997258309059999, 0.0003821581922423547, 0.9752022255856494, 0.024409114108818398, 0.6774590122476348, 0.32265420752811086, 0.8326398730899595, 0.16732096497331567, 0.9466073175599518, 0.053791961219623034, 0.9980770224660811, 0.0014402265836451387, 0.9994978020856828, 0.0012324263897480676, 0.014127513987633811, 0.9858986547084452, 0.0024896109906395246, 0.9983340072464493, 0.9985791403889751, 0.0013660453356894324, 0.9987686125733684, 0.0015654680447858439, 0.9970962045278153, 0.0027658701928649524, 0.014073411151927085, 0.985138780634896, 0.9993706324907479, 0.0006447552467682245, 0.050146353137141865, 0.9497415366882929, 0.9996979991794268, 0.00022279875176720008, 0.0014396847925436983, 0.9991412460253266, 0.9979180453328692, 0.0020920713738634576, 0.9924873910146471, 0.006989347824046811, 0.0011781099379501035, 0.9990372273816878, 0.002685226934983623, 0.9989044198139077, 0.998768948828339, 0.0015679261363082243, 0.0008496147560147584, 0.9991469530733559], "Term": ["access", "access", "accommodate", "accommodate", "air", "air", "airport", "airport", "amazing", "amazing", "answer", "answer", "apartment", "apartment", "area", "area", "arrive", "arrive", "bar", "bar", "bathroom", "bathroom", "beautiful", "beautiful", "bed", "bed", "bit", "bit", "book", "book", "building", "building", "bus", "bus", "care", "care", "central", "central", "cheap", "cheap", "check", "check", "chinatown", "chinatown", "clean", "clean", "close", "close", "come", "come", "comfortable", "comfortable", "convenient", "convenient", "day", "day", "distance", "distance", "door", "door", "easy", "easy", "enjoy", "enjoy", "environment", "environment", "experience", "experience", "family", "family", "feel", "feel", "first", "first", "floor", "floor", "food", "food", "friend", "friend", "friendly", "friendly", "get", "get", "give", "give", "go", "go", "good", "good", "great", "great", "help", "help", "helpful", "helpful", "home", "home", "hope", "hope", "hospitable", "hospitable", "hospitality", "hospitality", "host", "host", "hostel", "hostel", "hotel", "hotel", "house", "house", "information", "information", "kind", "kind", "kitchen", "kitchen", "landlord", "landlord", "little", "little", "location", "location", "love", "love", "lovely", "lovely", "machine", "machine", "make", "make", "mall", "mall", "meet", "meet", "min", "min", "minute", "minute", "money", "money", "need", "need", "next", "next", "nice", "nice", "night", "night", "noise", "noise", "noisy", "noisy", "option", "option", "owner", "owner", "price", "price", "public", "public", "question", "question", "recommend", "recommend", "reply", "reply", "respond", "respond", "response", "response", "restaurant", "restaurant", "road", "road", "room", "room", "shop", "shop", "shower", "shower", "sleep", "sleep", "small", "small", "space", "space", "staff", "staff", "station", "station", "stop", "stop", "store", "store", "street", "street", "time", "time", "tip", "tip", "toilet", "toilet", "train", "train", "transport", "transport", "trip", "trip", "value", "value", "visit", "visit", "walk", "walk", "warm", "warm", "washing", "washing", "water", "water", "welcome", "welcome", "welcoming", "welcoming", "window", "window", "wonderful", "wonderful"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [2, 1]};

    function LDAvis_load_lib(url, callback){
      var s = document.createElement('script');
      s.src = url;
      s.async = true;
      s.onreadystatechange = s.onload = callback;
      s.onerror = function(){console.warn("failed to load library " + url);};
      document.getElementsByTagName("head")[0].appendChild(s);
    }

    if(typeof(LDAvis) !== "undefined"){
      // already loaded: just create the visualization
      !function(LDAvis){
          new LDAvis("#" + "ldavis_el1849622229530641444586857286", ldavis_el1849622229530641444586857286_data);
      }(LDAvis);
    }else if(typeof define === "function" && define.amd){
      // require.js is available: use it to load d3/LDAvis
      require.config({paths: {d3: "https://d3js.org/d3.v5"}});
      require(["d3"], function(d3){
          window.d3 = d3;
          LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
            new LDAvis("#" + "ldavis_el1849622229530641444586857286", ldavis_el1849622229530641444586857286_data);
          });
        });
    }else{
        // require.js not available: dynamically load d3 & LDAvis
        LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
            LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                    new LDAvis("#" + "ldavis_el1849622229530641444586857286", ldavis_el1849622229530641444586857286_data);
                })
            });
    }
    </script>""" 
    st.components.v1.html(string, width=1250, height=900, scrolling=True)

    st.write("")
    st.write("")
    st.write("#### The topics :arrow_heading_down:")
    st.write("")
    st.dataframe(data = topics, use_container_width=True, column_config={"0": st.column_config.Column("Topic No.", width = "small"), "1": st.column_config.Column("Tokens", width = "Large")})


    # if not openai_api_key.startswith('sk-'):
    #     st.warning('Please enter your OpenAI API key!', icon='⚠')
    # if openai_api_key.startswith('sk-'):
    generate_response(response.choices[0].message.content)
    # message = st.chat_message("assistant", avatar = "🍀")
    # message.write("Hello human")

# 🍀 	🍈


st.title("Singapore:grey['s] Positive :grey[Reviews]	:cityscape:")
st.write("")
st.write("")

tab1, tab2 = st.tabs(["**BERT**", "**LDA**"])

with tab1:
   sgPosBERT()

with tab2:
   sgPosLDA()

