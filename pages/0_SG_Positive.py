from typing import Any
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from bertopic import BERTopic
from gensim.models import LdaMulticore
from openai import OpenAI
import json


st.set_page_config(
        page_title="SG_Positive",
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
TopicForOA = json.dumps(lda.print_topics())

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

response1 = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with a list of topics broken down into tokens for each topic . These topics is the result of topic modelling on Airbnb listings' reviews. Your job is to summarise these output from BERTopic's topic model, and similar topics can integrate them together into one. \n Then output them into comprehensible points. \n For example: '1. Many reviews discuss the quality and condition of the room, host responsiveness, and the apartment's location. 2. Noise levels and sound disturbance are a common concern among guests. 3. The proximity to public transportation, such as the BTS station, is frequently mentioned.' \n From the summarisation, give me a few recommendations for Airbnb Host to implement for them to improve their services."
    },
    {
      "role": "user",
      "content": p.to_json()
    }
  ],
  temperature=0.7,
  max_tokens=450,
  top_p=1
)

response2 = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with a list of topics broken down into tokens for each topic . These topics is the result of topic modelling on Airbnb listings' reviews. Your job is to summarise these output from LDA's topic model, and similar topics can integrate them together into one. \n Then output them into comprehensible points.\n For example: '1. Many reviews discuss the quality and condition of the room, host responsiveness, and the apartment's location. 2. Noise levels and sound disturbance are a common concern among guests. 3. The proximity to public transportation, such as the BTS station, is frequently mentioned.'"
    },
    {
      "role": "user",
      "content": TopicForOA
    }
  ],
  temperature=0.7,
  max_tokens=450,
  top_p=1
)


@st.cache_data(ttl=3600)
def generate_response(t):
    message = st.chat_message("assistant")
    message.write("Chat-GPT 4.0 says: ")
    st.info(t)

@st.cache_data(ttl=3600)
def sgPosBERT() -> None:

    st.write("### using BERT	:arrow_heading_down:")

    st.plotly_chart(r, use_container_width=True, sharing="streamlit", theme="streamlit")
    st.write("")
    st.write("")
    st.subheader("Topic Information :arrow_heading_down:")
    st.write("")
    st.dataframe(data=p, 
                #  width=None, height=None, 
                 use_container_width=True, hide_index=True, column_order=("Topic", "Count", "Name", "Representation"), 
                 column_config=None
                )
    generate_response(response1.choices[0].message.content)

@st.cache_data(ttl=3600)
def sgPosLDA() -> None:

    st.write("### using LDA :arrow_heading_down:")
    st.write("")
    string = """
<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css">


<div id="ldavis_el1857621416719624321706457361" style="background-color:white;"></div>
<script type="text/javascript">

var ldavis_el1857621416719624321706457361_data = {"mdsDat": {"x": [0.17087785914026832, -0.17087785914026832], "y": [0.0, 0.0], "topics": [1, 2], "cluster": [1, 1], "Freq": [56.31149409077173, 43.68850590922827]}, "tinfo": {"Term": ["room", "host", "apartment", "great", "small", "night", "bed", "bathroom", "good", "bus", "price", "hotel", "value", "bit", "family", "kitchen", "home", "amazing", "space", "money", "hostel", "house", "shower", "little", "food", "floor", "water", "air", "pool", "lovely", "host", "home", "family", "wonderful", "lovely", "beautiful", "amazing", "transport", "transportation", "shopping", "hospitality", "fantastic", "local", "mall", "accessible", "swimming", "tip", "welcoming", "hospitable", "orchard", "cosy", "visit", "neighborhood", "condo", "awesome", "thoughtful", "attentive", "enjoyable", "prompt", "neighbourhood", "apartment", "bus", "stop", "pool", "response", "centre", "quick", "public", "house", "distance", "city", "great", "airport", "food", "helpful", "responsive", "station", "close", "convenient", "friendly", "location", "nice", "easy", "perfect", "quiet", "clean", "minute", "restaurant", "time", "comfortable", "excellent", "experience", "good", "lot", "area", "small", "bathroom", "bed", "price", "bit", "hotel", "value", "shower", "hostel", "money", "kitchen", "water", "air", "toilet", "floor", "door", "window", "luggage", "noise", "towel", "wifi", "machine", "free", "issue", "worth", "washing", "hot", "noisy", "tv", "light", "night", "coffee", "room", "problem", "bedroom", "morning", "space", "little", "good", "check", "day", "clean", "breakfast", "people", "big", "location", "staff", "nice", "comfortable", "great", "time", "area", "thing"], "Freq": [10784.0, 6830.0, 4918.0, 12648.0, 2664.0, 2330.0, 2223.0, 2212.0, 10906.0, 2781.0, 1678.0, 1583.0, 1543.0, 1483.0, 1753.0, 1193.0, 1643.0, 1585.0, 1754.0, 1067.0, 1006.0, 2409.0, 979.0, 1676.0, 2934.0, 875.0, 874.0, 864.0, 1405.0, 1210.0, 6829.06343531424, 1642.5366371698346, 1752.5474126042461, 1143.5105652577975, 1210.1469523605926, 999.4656324484254, 1584.3594828526402, 708.5415056576186, 824.1848422480912, 804.6908730620054, 640.4504732166537, 783.0174317584883, 1041.7829859716621, 774.0477130992573, 545.4440196051743, 474.4260215129998, 388.61671815382266, 363.01298895844207, 321.779131769363, 425.68569927592296, 487.30364146047623, 320.03300215541515, 661.3688314438689, 273.5585403855677, 644.6924162489888, 272.0028369778229, 286.6055242853312, 205.58021774473008, 255.11776030007246, 301.33631695825363, 4897.689286182303, 2758.899969122724, 936.3144243962724, 1390.3912524347954, 558.9813185138746, 520.8028124923178, 1009.6171804071946, 954.6706243786782, 2279.7433195373164, 1310.8333890347737, 1752.0459934324224, 11021.91954781437, 1208.2495667297076, 2665.2533679726607, 3233.9869939466503, 1258.7526238885355, 3203.4972038667097, 2149.603956726881, 3131.891652935549, 3130.2419206989553, 7659.448628184902, 5817.499460844967, 2801.8412951017553, 1947.9564336249152, 1796.2688824308648, 6206.531871995667, 1987.1952853291068, 1788.7113280462825, 2707.2055065972813, 2714.1960355632596, 1358.2284011888942, 1512.616505171623, 3961.331260118915, 1641.2411175420245, 1534.7212260492797, 2663.9480913143393, 2211.6418583470136, 2222.5402457007135, 1677.6847397677057, 1482.527967898044, 1583.0942618430227, 1542.675779632319, 979.1780956591696, 1005.6050054594293, 1067.1433910359046, 1193.0601923054737, 873.7293810862451, 864.0521916656949, 742.5307861283324, 875.0728489826711, 816.3838198677828, 642.6716346277192, 671.2187862087488, 587.0423184755556, 550.2078795031911, 631.9419908884956, 539.675669744746, 684.4006263801712, 652.7864509304032, 596.297609341504, 487.08298780335053, 478.04105683776703, 435.46953256525967, 434.30520162537255, 485.0219446020855, 2322.120634330948, 533.066887337315, 9737.417375621748, 813.3062841017493, 545.9256249813465, 653.2967521701536, 1561.9054383686682, 1453.4024248553103, 6944.698700514234, 1591.987168881161, 1676.982418569468, 3633.856985626929, 880.3687902621114, 1042.8463655086705, 916.9731135278961, 2829.8018034395745, 1322.8023757486528, 1924.2785171708595, 1256.9398856532064, 1626.6658390213706, 1149.3263525320344, 1022.6433683122385, 875.6256325141221], "Total": [10784.0, 6830.0, 4918.0, 12648.0, 2664.0, 2330.0, 2223.0, 2212.0, 10906.0, 2781.0, 1678.0, 1583.0, 1543.0, 1483.0, 1753.0, 1193.0, 1643.0, 1585.0, 1754.0, 1067.0, 1006.0, 2409.0, 979.0, 1676.0, 2934.0, 875.0, 874.0, 864.0, 1405.0, 1210.0, 6830.083520426733, 1643.2719090043101, 1753.6148445550134, 1144.209767203029, 1210.9089939048538, 1000.1995145006618, 1585.611728209185, 709.2543704759382, 825.0421520295479, 805.5291545742524, 641.1499735567612, 783.8934461749811, 1043.0356745210102, 774.982516091742, 546.2915889473763, 475.2669634942015, 389.36272772713687, 363.7645524121735, 322.4800904314534, 426.664149272648, 488.60369807380823, 320.89470287999063, 663.1946511839473, 274.31718805474094, 646.5753226491215, 272.86368828811754, 287.52986338921517, 206.2760864046248, 255.9967575847431, 302.3916090384398, 4918.167979867419, 2781.718687465694, 942.2088275503447, 1405.0996493002713, 561.5989519810141, 524.0680505217119, 1022.9735919237485, 975.3225112043461, 2409.282151117687, 1370.6088484220934, 1854.6546383703176, 12648.58538683574, 1269.913398561734, 2934.7251438050116, 3712.320447506551, 1353.3061224602334, 3764.701081198485, 2436.3962568657553, 3723.4396630280894, 3779.804262077355, 10489.250431624478, 7741.777978015826, 3475.0771800388657, 2321.0982793931053, 2126.169228547619, 9840.388857622596, 2427.4322794650498, 2138.204526854619, 3856.5318591293158, 3971.135921216466, 1512.9315491009734, 1811.0096126584622, 10906.029960633148, 2219.6818982228924, 2557.3645943615184, 2664.6798995813333, 2212.3376344928497, 2223.258482716022, 1678.3824172288462, 1483.2715618029442, 1583.9564377735246, 1543.589893410419, 979.7603950718967, 1006.2146807682858, 1067.8088053804295, 1193.8243098005685, 874.3422666398704, 864.7194350493778, 743.1127257520949, 875.7697937835517, 817.1537404459959, 643.2986929715271, 671.9420288819177, 587.6804616820818, 550.8240121111847, 632.6657312871423, 540.3235754042538, 685.2445259465339, 653.6156353619019, 597.1128755659697, 487.749409586822, 478.7027037695163, 436.07343568121576, 434.9102428046112, 485.71676280296373, 2330.4774942478703, 536.1273677733124, 10784.230853000288, 835.0130554033578, 549.6176978581955, 664.9281765573904, 1754.0938064575566, 1676.4544443982138, 10906.029960633148, 2197.56242078897, 2524.3068591143488, 9840.388857622596, 1062.8011325244138, 1394.1563053640764, 1222.3813728596103, 10489.250431624478, 2595.2618153077565, 7741.777978015826, 3971.135921216466, 12648.58538683574, 3856.5318591293158, 2557.3645943615184, 1139.431457947359], "Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2"], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -3.4668, -4.8918, -4.827, -5.2539, -5.1973, -5.3886, -4.9278, -5.7326, -5.5814, -5.6053, -5.8336, -5.6326, -5.3471, -5.6441, -5.9942, -6.1337, -6.3332, -6.4013, -6.5219, -6.2421, -6.1069, -6.5274, -5.8015, -6.6843, -5.827, -6.69, -6.6377, -6.9699, -6.7541, -6.5876, -3.7993, -4.3732, -5.4538, -5.0584, -5.9697, -6.0404, -5.3785, -5.4344, -4.564, -5.1174, -4.8272, -2.9881, -5.1989, -4.4077, -4.2143, -5.1579, -4.2238, -4.6227, -4.2464, -4.2469, -3.3521, -3.6272, -4.3577, -4.7212, -4.8023, -3.5624, -4.7013, -4.8065, -4.3921, -4.3895, -5.0818, -4.9742, -4.0114, -4.8926, -4.9597, -4.1544, -4.3405, -4.3356, -4.6168, -4.7405, -4.6748, -4.7007, -5.1553, -5.1286, -5.0692, -4.9577, -5.2692, -5.2803, -5.4319, -5.2677, -5.3371, -5.5763, -5.5329, -5.6669, -5.7317, -5.5932, -5.751, -5.5134, -5.5607, -5.6512, -5.8535, -5.8723, -5.9655, -5.9682, -5.8578, -4.2917, -5.7633, -2.8582, -5.3409, -5.7395, -5.5599, -4.6883, -4.7603, -3.1962, -4.6692, -4.6172, -3.8439, -5.2616, -5.0923, -5.2209, -4.094, -4.8545, -4.4797, -4.9055, -4.6477, -4.995, -5.1118, -5.267], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5741, 0.5738, 0.5737, 0.5737, 0.5736, 0.5735, 0.5735, 0.5733, 0.5732, 0.5732, 0.5732, 0.5732, 0.5731, 0.5731, 0.5727, 0.5725, 0.5724, 0.5722, 0.5721, 0.572, 0.5716, 0.5716, 0.5715, 0.5715, 0.5714, 0.5711, 0.5711, 0.5709, 0.5708, 0.5708, 0.5701, 0.566, 0.568, 0.5637, 0.5696, 0.568, 0.5611, 0.5529, 0.519, 0.5297, 0.5174, 0.4366, 0.5245, 0.478, 0.4363, 0.5018, 0.4128, 0.449, 0.4013, 0.3857, 0.2599, 0.2885, 0.3589, 0.399, 0.4057, 0.1134, 0.3742, 0.3958, 0.2204, 0.1937, 0.4664, 0.3942, -0.4385, 0.2724, 0.0636, 0.8278, 0.8278, 0.8278, 0.8277, 0.8276, 0.8275, 0.8275, 0.8275, 0.8275, 0.8275, 0.8274, 0.8274, 0.8273, 0.8273, 0.8273, 0.8271, 0.8271, 0.827, 0.827, 0.827, 0.8269, 0.8269, 0.8269, 0.8268, 0.8267, 0.8267, 0.8267, 0.8267, 0.8267, 0.8267, 0.8245, 0.8224, 0.726, 0.8017, 0.8213, 0.8104, 0.712, 0.6853, 0.3767, 0.5057, 0.4191, -0.1681, 0.6398, 0.5377, 0.5406, -0.4821, 0.1542, -0.564, -0.3223, -1.2229, -0.3825, -0.0885, 0.5647]}, "token.table": {"Topic": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], "Freq": [0.997635715113489, 0.001830524247914659, 0.0011564444598644846, 0.9991680133229147, 0.9512459679283206, 0.04882222683075818, 0.9989835290818608, 0.0006306714198749121, 0.9958992901523542, 0.004066554880164778, 0.6002272821733634, 0.400021178933779, 0.9981571883248248, 0.0034778996108878915, 0.9975635898959657, 0.00309322043378594, 0.0004520105721698475, 0.9998473856397028, 0.9988007247721364, 0.000999800525297434, 0.0004497902550576844, 0.9998837369932324, 0.0072777860239719265, 0.993417792272168, 0.24951296442491605, 0.7501750438611411, 0.0006741853789635671, 0.99981691700297, 0.1712455834213361, 0.8280006231361307, 0.9918328594591309, 0.008268269578673437, 0.9941457020349597, 0.005724447420546792, 0.2757600850229472, 0.7244390352418019, 0.9446502673616258, 0.05553594608347458, 0.6307677562144215, 0.369294349296473, 0.882450871421801, 0.117796930278166, 0.0055956852425942056, 0.9941667447675705, 0.6834316562925976, 0.31653411641849494, 0.9988437178982833, 0.003645415028825852, 0.8411577152972849, 0.15899277377266688, 0.996717793827328, 0.002046648447284041, 0.33553765341237846, 0.6643407848554411, 0.9565092196138105, 0.04377616565738263, 0.0012237599248511157, 0.9985880986785103, 0.8063130269724432, 0.19366476343770675, 0.9986615685344968, 0.004847871691915033, 0.8975951362815865, 0.10245010760209566, 0.8354455931236056, 0.1645490989760968, 0.9996493844945927, 0.0005702506471731846, 0.998860245382404, 0.001275683582863862, 0.0011418525816924352, 0.9991210089808807, 0.9080918550841528, 0.09166105404038916, 0.0014593330732831345, 0.998183822125664, 0.8280852083805453, 0.17196657682024105, 0.3631935740409469, 0.6368036787968634, 0.8714017941857245, 0.128630985224113, 0.8711532438349109, 0.12876043616360155, 0.999834531946405, 0.0006085420157920907, 0.9985112555915899, 0.0031009666322720183, 0.9982063891379707, 0.0015596974830280792, 0.9998413605890041, 0.00014641109395065224, 0.0009938237029462334, 0.9997866451639109, 0.002088979218470167, 0.9985320664287398, 0.0006313304937891107, 0.9993961716681622, 0.9463399705768326, 0.05395798077850361, 0.0015299511607403758, 0.9990581079634653, 0.0008376441925253245, 0.9993095216827121, 0.002058813029694964, 0.9985243194020575, 0.13301882478533372, 0.8667101005071296, 0.9990070574321576, 0.0009587399783418019, 0.730176102661117, 0.26980002226543426, 0.7392951221135816, 0.260397672505576, 0.9992493292977183, 0.0008258258919815854, 0.0014882236220049466, 0.9985980503653191, 0.0018507428613527184, 0.999401145130468, 0.9987322086997564, 0.0012903516908265586, 0.8185604256848266, 0.1812614933574855, 0.0009364972408555189, 0.9992425559928387, 0.01804706195807341, 0.9820609548851613, 0.9966907887751668, 0.003015705866188099, 0.9953979905630816, 0.003306970068315886, 0.7513777864101011, 0.2485217227184175, 0.0034327729058726184, 0.9963623359295275, 0.0017016049795798233, 0.9988421230133563, 0.0022931917383085757, 0.9975384061642304, 0.9984433909580165, 0.002343763828539945, 0.2517651705547738, 0.748122714782419, 0.8392578708512684, 0.16069978738579216, 0.9892536808276974, 0.010675399433392418, 0.0005958117707471495, 0.999772151313717, 0.026346893449914713, 0.9736374715809392, 0.9961063663690617, 0.003906299475957105, 0.9791632911463803, 0.02153133938646491, 0.9873177645775283, 0.012708050435156305, 0.8447116889311973, 0.1552087178993848, 0.9953722278650157, 0.005341890310545701, 0.9303142719189135, 0.07019845578419125, 0.8366832908317184, 0.16322105561781428, 0.09708620060824397, 0.902892392858139, 0.9993430969304493, 0.0012414199961868935, 0.0010206577087927891, 0.9992238969081406, 0.00037527959743199066, 0.9997448475588231, 0.10945822811366604, 0.8904882932997206, 0.4901239607107467, 0.5097751572486776, 0.8507979600282982, 0.14901581504086023, 0.9934103487795937, 0.006368015056279447, 0.9973342066848353, 0.0021040806048203274, 0.23169449830320257, 0.7688044716424449, 0.9968347261831132, 0.0036648335521437983, 0.7019260047319189, 0.2979360840180919, 0.9990684066519303, 0.0025682992458918516, 0.0013456908559706239, 0.9998483059861735, 0.001815461886215208, 0.9985040374183644, 0.9996413550814393, 0.001409931389395542, 0.9987368499573213, 0.001212059283928788, 0.002299325013711536, 0.9979070559508065, 0.0006478404686821268, 0.9996178431765216, 0.9972118490210004, 0.003116287028190626, 0.002050233132720983, 0.9984635356351187, 0.001143716869416638, 0.9996085438701416, 0.9978982217835586, 0.0027490309140042937, 0.0015806135065440095, 0.9989477361358139, 0.0015544878466654383, 0.9995356854058768, 0.999816670676093, 0.000873965621220361, 0.0016747252335702128, 0.9981362392078468], "Term": ["accessible", "accessible", "air", "air", "airport", "airport", "amazing", "amazing", "apartment", "apartment", "area", "area", "attentive", "attentive", "awesome", "awesome", "bathroom", "bathroom", "beautiful", "beautiful", "bed", "bed", "bedroom", "bedroom", "big", "big", "bit", "bit", "breakfast", "breakfast", "bus", "bus", "centre", "centre", "check", "check", "city", "city", "clean", "clean", "close", "close", "coffee", "coffee", "comfortable", "comfortable", "condo", "condo", "convenient", "convenient", "cosy", "cosy", "day", "day", "distance", "distance", "door", "door", "easy", "easy", "enjoyable", "enjoyable", "excellent", "excellent", "experience", "experience", "family", "family", "fantastic", "fantastic", "floor", "floor", "food", "food", "free", "free", "friendly", "friendly", "good", "good", "great", "great", "helpful", "helpful", "home", "home", "hospitable", "hospitable", "hospitality", "hospitality", "host", "host", "hostel", "hostel", "hot", "hot", "hotel", "hotel", "house", "house", "issue", "issue", "kitchen", "kitchen", "light", "light", "little", "little", "local", "local", "location", "location", "lot", "lot", "lovely", "lovely", "luggage", "luggage", "machine", "machine", "mall", "mall", "minute", "minute", "money", "money", "morning", "morning", "neighborhood", "neighborhood", "neighbourhood", "neighbourhood", "nice", "nice", "night", "night", "noise", "noise", "noisy", "noisy", "orchard", "orchard", "people", "people", "perfect", "perfect", "pool", "pool", "price", "price", "problem", "problem", "prompt", "prompt", "public", "public", "quick", "quick", "quiet", "quiet", "response", "response", "responsive", "responsive", "restaurant", "restaurant", "room", "room", "shopping", "shopping", "shower", "shower", "small", "small", "space", "space", "staff", "staff", "station", "station", "stop", "stop", "swimming", "swimming", "thing", "thing", "thoughtful", "thoughtful", "time", "time", "tip", "tip", "toilet", "toilet", "towel", "towel", "transport", "transport", "transportation", "transportation", "tv", "tv", "value", "value", "visit", "visit", "washing", "washing", "water", "water", "welcoming", "welcoming", "wifi", "wifi", "window", "window", "wonderful", "wonderful", "worth", "worth"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [2, 1]};

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
       new LDAvis("#" + "ldavis_el1857621416719624321706457361", ldavis_el1857621416719624321706457361_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://d3js.org/d3.v5"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
        new LDAvis("#" + "ldavis_el1857621416719624321706457361", ldavis_el1857621416719624321706457361_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://d3js.org/d3.v5.js", function(){
         LDAvis_load_lib("https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el1857621416719624321706457361", ldavis_el1857621416719624321706457361_data);
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

    generate_response(response2.choices[0].message.content)


st.title("Singapore:grey['s] Positive :grey[Reviews]	:cityscape:")
st.write("")
st.write("")

tab1, tab2 = st.tabs(["**BERT**", "**LDA**"])

with tab1:
   sgPosBERT()

with tab2:
   sgPosLDA()

