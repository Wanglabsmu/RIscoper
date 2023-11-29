
## 1. Overview
**RIscoper 2.0** is a BERT-based deep learning tool for scanning the RNA biomedical relations in literatures, which recruited over 35,000 sentences manually curated from more than 15,000 biomedical literatures. Benefit from approximately 100,000 annotated named entities, the named entity recognition (NER) and relation extraction (RE) tasks were integrated. Both the online tools and data resources of RIscoper 2.0 are available at http://www.rnainter.org/riscoper.

<u>**Conda install***</u> 

```{r}
conda create -n RIscoper
conda activate RIscoper
pip install -r requirements.txt 
```
<u>**Run in python***</u> 

```{r}
import os
os.mkdir("RIscoper")
```
Download all files in the github and ensure the file structure is as follow:

├── RIscoper
│   ├── code

│   ├── data

│   └── model_weights

```{r}
import os
os.chdir("RIscoper")
sys.path.append(os.getcwd() + r"\\code")
from A5_eval import * 
```

## 2. Running
RIscoper 2.0 allows for three types of inputs, including single sentence, multiple sentences and sentences labeled with BIO (Begin, Inside, Outside) tags.

### 2.1 Single sentence

Input:

```{r}
your_text = "taken together, our data suggest that mir-155 may attenuate the expression of smad2 by directly targeting the 3'utr of smad2."
```
Result of RE:
```{r}
predict_single(your_text)[1] 
```
Result of NER:
```{r}
NER.recognize(your_text)
```

### 2.2 Multiple sentences

Input:

```{r}
data_path = "data/example_sentences.txt"
test_data, _ = load_data3(data_path)
```
Result of RE:
```{r}
pred_y_list = t_map(predict_single, test_data)
pred_y_list = np.array([i[1] for i in pred_y_list])  # ['RDI', 'RDI', 'RDI', 'NEG', 'RDI']
```
Result of NER:
```{r}
y_pred = t_map(NER.recognize, test_data)  # [[('circβ-catenin', 'rna'), ('NSCLC', 'NEG')]]
```
### 2.3 Sentences labeled with BIO tags

Input:

```{r}
data_path = "data/example_BIO.txt"
test_data, _ = load_data(data_path)
```
Result of RE:
```{r}
test_data = [" ".join([i[0] for i in j]) for j in test_data]
pred_y_list = t_map(predict_single, test_data)
pred_y_list = np.array([i[1] for i in pred_y_list])  # ['RDI', '', 'RDI', 'RRI', 'RRI']
```
Result of NER:
```{r}
y_pred = predict_NER(test_data)  # [['b-rna', 'o', 'b-NEG', 'o', ', 'o']]
```

## 3. Data availability
Both the online tools and data resources of RIscoper 2.0 are available at http://www.rnainter.org/riscoper.

## 4. Contact email
Please don't hesitate to address comments/questions/suggestions regarding this tool to: wangdong79@smu.edu.cn
