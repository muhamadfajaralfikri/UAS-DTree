import streamlit as st
import numpy as np
import pandas as pd
import seaborn  as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
st.set_option('deprecation.showPyplotGlobalUse', False)

dataset_encoded = pd.read_csv('jamur.csv')

# Menggunakan .iloc untuk mengakses baris dan kolom berdasarkan indeks
X = dataset_encoded.drop(dataset_encoded.columns[[1, 2]], axis=1)
y = dataset_encoded['class']

X = pd.get_dummies(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, y_train)

# Konten utama
st.title("Klasifikasi Jamur")
st.write("#### Nama : Muhamad Fajar Al Fikri \n#### NIM : 211351087 \n#### Malam B")
st.markdown("----")

st.write('Aplikasi ini dibuat untuk memberikan layanan yang dapat membantu pengguna dalam mengidentifikasi apakah suatu jenis jamur yang ditemukan aman untuk dikonsumsi atau berpotensi beracun. Aplikasi ini dapat menjadi alat yang berguna untuk pecinta jamur atau peneliti yang ingin dengan cepat menentukan karakteristik keamanan jamur berdasarkan fitur-fitur tertentu.')

cs = st.selectbox(
    'Cap Shape', ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'])
if cs == 'bell':
    cs = 0
elif cs == 'conical':
    cs = 1
elif cs == 'convex':
    cs = 5
elif cs == 'flat':
    cs = 2
elif cs == 'knobbed':
    cs = 3
elif cs == 'sunken':
    cs = 4

csu = st.selectbox(
    'Cap Surface', ['fibrous', 'grooves', 'scaly', 'smooth'])
if csu == 'fibrous':
    csu = 0
elif csu == 'grooves':
    csu = 1
elif csu == 'scaly':
    csu = 3
elif csu == 'smooth':
    csu = 2

cc = st.selectbox(
    'Cap Color', ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'])
if cc == 'brown':
    cc = 4
elif cc == 'buff':
    cc = 0
elif cc == 'cinnamon':
    cc = 1
elif cc == 'gray':
    cc = 3
elif cc == 'green':
    cc = 6
elif cc == 'pink':
    cc = 5
elif cc == 'purple':
    cc = 7
elif cc == 'red':
    cc = 2
elif cc == 'white':
    cc = 8
elif cc == 'yellow':
    cc = 9

bruises = st.selectbox(
    'Bruises', ['Bruises', 'No'])
if bruises == 'Bruises':
    bruises = 1
else:
    bruises = 0

odor = st.selectbox(
    'Odor', ['almond' , 'anise' , 'creosote' , 'fishy' , 'foul' , 'musty' , 'none' , 'pungent' , 'spicy' ])
if odor == 'almond':
    odor = 0
elif odor == 'anise':
    odor = 3
elif odor == 'creosote':
    odor = 1
elif odor == 'fishy':
    odor = 8
elif odor == 'foul':
    odor = 2
elif odor == 'musty':
    odor = 4
elif odor == 'none':
    odor = 5
elif odor == 'pungent':
    odor = 6
elif odor == 'spicy':
    odor = 7

gill_attachment = st.selectbox(
    'Gill Attachment', ['attached', 'Nothced'])
if gill_attachment == 'attached':
    gill_attachment = 0
else:
    gill_attachment = 1

gill_spacing = st.selectbox(
    'Gill Spacing', ['close', 'crowded'])
if gill_spacing == 'close':
    gill_spacing = 0
else:
    gill_spacing = 1

gill_size = st.selectbox(
    'Gill Size', ['broad', 'narrow'])
if gill_size == 'broad':
    gill_size = 0
else:
    gill_size = 1

gill_color = st.selectbox(
    'Gill Color', ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple' 'red' 'white', 'yellow'])
if gill_color == 'black':
    gill_color = 4
elif gill_color == 'brown':
    gill_color = 5
elif gill_color == 'buff':
    gill_color = 0
elif gill_color == 'chocolate':
    gill_color = 3
elif gill_color == 'gray':
    gill_color = 2
elif gill_color == 'green':
    gill_color = 8
elif gill_color == 'orange':
    gill_color = 6
elif gill_color == 'pink':
    gill_color = 7
elif gill_color == 'purple':
    gill_color = 9
elif gill_color == 'red':
    gill_color = 1
elif gill_color == 'white':
    gill_color = 10
elif gill_color == 'yellow':
    gill_color = 11

stalk_shape = st.selectbox(
    'Stalk Shape', ['enlarging', 'tapering'])
if stalk_shape == 'enlarging':
    stalk_shape = 0
else:
    stalk_shape = 1

stalk_root = st.selectbox(
    'Stalk Root', ['bulbous', 'club', 'equal', 'rooted', 'missing'])
if stalk_root == 'bulbous':
    stalk_root = 1
elif stalk_root == 'club':
    stalk_root = 2
elif stalk_root == 'equal':
    stalk_root = 3
elif stalk_root == 'rooted':
    stalk_root = 4
elif stalk_root == 'missing':
    stalk_root = 0

stalk_surface_above_ring = st.selectbox(
    'Stalk Surface Above Ring', ['fibrous', 'scaly', 'silky', 'smooth'])
if stalk_surface_above_ring == 'fibrous':
    stalk_surface_above_ring = 0
elif stalk_surface_above_ring == 'scaly':
    stalk_surface_above_ring = 3
elif stalk_surface_above_ring == 'silky':
    stalk_surface_above_ring = 1
elif stalk_surface_above_ring == 'smooth':
    stalk_surface_above_ring = 2

stalk_surface_below_ring = st.selectbox(
    'Stalk Surface Below Ring', ['fibrous', 'scaly', 'silky', 'smooth'])
if stalk_surface_below_ring == 'fibrous':
    stalk_surface_below_ring = 0
elif stalk_surface_below_ring == 'scaly':
    stalk_surface_below_ring = 3
elif stalk_surface_below_ring == 'silky':
    stalk_surface_below_ring = 1
elif stalk_surface_below_ring == 'smooth':
    stalk_surface_below_ring = 2

stalk_color_above_ring = st.selectbox(
    'Stalk Color Above Ring', ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
if stalk_color_above_ring == 'brown':
    stalk_color_above_ring = 4
elif stalk_color_above_ring == 'buff':
    stalk_color_above_ring = 0
elif stalk_color_above_ring == 'cinnamon':
    stalk_color_above_ring = 1
elif stalk_color_above_ring == 'gray':
    stalk_color_above_ring = 3
elif stalk_color_above_ring == 'orange':
    stalk_color_above_ring = 5
elif stalk_color_above_ring == 'pink':
    stalk_color_above_ring = 6
elif stalk_color_above_ring == 'red':
    stalk_color_above_ring = 2
elif stalk_color_above_ring == 'white':
    stalk_color_above_ring = 7
elif stalk_color_above_ring == 'yellow':
    stalk_color_above_ring = 8

stalk_color_below_ring = st.selectbox(
    'Stalk Color Below Ring', ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
if stalk_color_below_ring == 'brown':
    stalk_color_below_ring = 4
elif stalk_color_below_ring == 'buff':
    stalk_color_below_ring = 0
elif stalk_color_below_ring == 'cinnamon':
    stalk_color_below_ring = 1
elif stalk_color_below_ring == 'gray':
    stalk_color_below_ring = 3
elif stalk_color_below_ring == 'orange':
    stalk_color_below_ring = 5
elif stalk_color_below_ring == 'pink':
    stalk_color_below_ring = 6
elif stalk_color_below_ring == 'red':
    stalk_color_below_ring = 2
elif stalk_color_below_ring == 'white':
    stalk_color_below_ring = 7
elif stalk_color_below_ring == 'yellow':
    stalk_color_below_ring = 8

veil_color = st.selectbox(
    'Veil Color', ['brown', 'orange', 'white', 'yellow'])
if veil_color == 'brown':
    veil_color = 0
elif veil_color == 'orange':
    veil_color = 1
elif veil_color == 'white':
    veil_color = 2
elif veil_color == 'yellow':
    veil_color = 3

ring_number = st.selectbox(
    'Ring Number', ['none', 'one', 'two'])
if ring_number == 'none':
    ring_number = 0
elif ring_number == 'one':
    ring_number = 1
elif ring_number == 'two':
    ring_number = 2

ring_type = st.selectbox(
    'Ring Type', ['evanescent', 'flaring', 'large', 'none', 'pendant'])
if ring_type == 'evanescent':
    ring_type = 0
elif ring_type == 'flaring':
    ring_type = 1
elif ring_type == 'large':
    ring_type = 2
elif ring_type == 'none':
    ring_type = 3
elif ring_type == 'pendant':
    ring_type = 4

spore_print_color = st.selectbox(
    'Spore Print Color', ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'])
if spore_print_color == 'black':
    spore_print_color = 2
elif spore_print_color == 'brown':
    spore_print_color = 3
elif spore_print_color == 'buff':
    spore_print_color = 0
elif spore_print_color == 'chocolate':
    spore_print_color = 1
elif spore_print_color == 'green':
    spore_print_color = 5
elif spore_print_color == 'orange':
    spore_print_color = 4
elif spore_print_color == 'purple':
    spore_print_color = 6
elif spore_print_color == 'white':
    spore_print_color = 7
elif spore_print_color == 'yellow':
    spore_print_color = 8

population = st.selectbox(
    'Population', ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'])
if population == 'abundant':
    population = 0
elif population == 'clustered':
    population = 1
elif population == 'numerous':
    population = 2
elif population == 'scattered':
    population = 3
elif population == 'several':
    population = 4
elif population == 'solitary':
    population = 5

habitat = st.selectbox(
    'Habitat', ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'])
if habitat == 'grasses':
    habitat = 1
elif habitat == 'leaves':
    habitat = 2
elif habitat == 'meadows':
    habitat = 3
elif habitat == 'paths':
    habitat = 4
elif habitat == 'urban':
    habitat = 5
elif habitat == 'waste':
    habitat = 6
elif habitat == 'woods':
    habitat = 0

st.write('Veil Type')
{'partial'}
veil_type = 0

predict = ''

if st.button('Klasifikasi Jamur'):
    data = np.array([[cs, csu, cc, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_type, veil_color, ring_number, ring_type, spore_print_color, population, habitat]])
    prediction = clf_en.predict(data)

    if (prediction[0] == 0):
        predict = 'Jamur dapat dimakan atau aman untuk dikonsumsi (Edible)'
    else:
        predict = 'Jamur beracun dan tidak aman untuk dikonsumsi (Poisonous)'

st.write(predict)


st.markdown("----")
st.title("Visualisasi DTree")
plt.figure(figsize=(12, 8))
plot_tree(clf_en.fit(X_train, y_train))
st.pyplot()  # Use st.pyplot() without passing the figure

y_pred_en = clf_en.predict(X_test)
y_pred_train_en = clf_en.predict(X_train)

accuracy_test = accuracy_score(y_test, y_pred_en) * 100
accuracy_train = accuracy_score(y_train, y_pred_train_en) * 100

st.write('Nilai akurasi model menggunakan Criterion Entropy, mengindikasikan sejauh mana model tersebut berhasil dalam melakukan prediksi dengan benar. Dalam hal ini, nilai akurasi sebesar {0:0.2f}% menunjukkan bahwa model Decision Tree dapat memprediksi kelas jamur dengan tingkat keakuratan sekitar {0:0.2f}%'.format(accuracy_test))
st.write('nilai akurasi Training-set didapatkan sebesar {0:0.2f}% menunjukkan bahwa model Decision Tree (atau model klasifikasi jamur) berhasil memprediksi kelas dengan benar untuk sekitar {0:0.2f}% dari data pada saat proses pelatihan.'.format(accuracy_train))

st.markdown("----")

st.title("Confussion Matrix")

cm = confusion_matrix(y_test, y_pred_en)
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, linewidths=0.5,linecolor="red", fmt= '.0f',ax=ax)
st.pyplot(f)

# # Kolom dan nilai mapping
# columns_mapping = {
#     'cap-shape': {'bell': 0, 'conical': 1, 'convex': 5, 'flat': 2, 'knobbed': 3, 'sunken': 4},
#     'cap-surface': {'fibrous': 0, 'grooves': 1, 'scaly': 3, 'smooth': 2},
#     'cap-color': {'brown': 4, 'buff': 0, 'cinnamon': 1, 'gray': 3, 'green': 6, 'pink': 5, 'purple': 7, 'red': 2, 'white': 8, 'yellow': 9},
#     'bruises': {'bruises': 1, 'no': 0},
#     'odor': {'almond': 0, 'anise': 3, 'creosote': 1, 'fishy': 8, 'foul': 2, 'musty': 4, 'none': 5, 'pungent': 6, 'spicy': 7},
#     'gill-attachment': {'attached': 0, 'notched': 1},
#     'gill-spacing': {'close': 0, 'crowded': 1},
#     'gill-size': {'broad': 0, 'narrow': 1},
#     'gill-color': {'black': 4, 'brown': 5, 'buff': 0, 'chocolate': 3, 'gray': 2, 'green': 8, 'orange': 6, 'pink': 7, 'purple': 9, 'red': 1, 'white': 10, 'yellow': 11},
#     'stalk-shape': {'enlarging': 0, 'tapering': 1},
#     'stalk-root': {'bulbous': 1, 'club': 2, 'equal': 3, 'rooted': 4, 'missing': 0},
#     'stalk-surface-above-ring': {'fibrous': 0, 'scaly': 3, 'silky': 1, 'smooth': 2},
#     'stalk-surface-below-ring': {'fibrous': 0, 'scaly': 3, 'silky': 1, 'smooth': 2},
#     'stalk-color-above-ring': {'brown': 4, 'buff': 0, 'cinnamon': 1, 'gray': 3, 'orange': 5, 'pink': 6, 'red': 2, 'white': 7, 'yellow': 8},
#     'stalk-color-below-ring': {'brown': 4, 'buff': 0, 'cinnamon': 1, 'gray': 3, 'orange': 5, 'pink': 6, 'red': 2, 'white': 7, 'yellow': 8},
#     'veil-type': {'partial': 0},
#     'veil-color': {'brown': 0, 'orange': 1, 'white': 2, 'yellow': 3},
#     'ring-number': {'none': 0, 'one': 1, 'two': 2},
#     'ring-type': {'evanescent': 0, 'flaring': 1, 'large': 2, 'none': 3, 'pendant': 4},
#     'spore-print-color': {'black': 2, 'brown': 3, 'buff': 0, 'chocolate': 1, 'green': 5, 'orange': 4, 'purple': 6, 'white': 7, 'yellow': 8},
#     'population': {'abundant': 0, 'clustered': 1, 'numerous': 2, 'scattered': 3, 'several': 4, 'solitary': 5},
#     'habitat': {'grasses': 1, 'leaves': 2, 'meadows': 3, 'paths': 4, 'urban': 5, 'waste': 6, 'woods': 0},
# }

# # Streamlit App
# st.title("Predict Mushroom Edibility")
# st.sidebar.title("Select Mushroom Features")

# # Create selectbox for each column
# selected_features = {}
# for column, options in columns_mapping.items():
#     selected_features[column] = st.sidebar.selectbox(f'Select {column}', list(options.keys()))

# # Add a button to trigger prediction
# if st.sidebar.button('Predict'):
#     # Prepare the data for prediction
#     input_data = [selected_features[column] for column in columns_mapping.keys()]
#     data = np.array([input_data])

#     # Konversi nilai string ke nilai numerik
#     for i, column in enumerate(columns_mapping.keys()):
#         data[0, i] = columns_mapping[column][input_data[i]]

#     # Prediction
#     prediction = clf_en.predict(data)

#     # Display prediction result
#     st.write(f"The predicted mushroom edibility is: {'Edible' if prediction[0] == 0 else 'Poisonous'}")

