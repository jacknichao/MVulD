
## Dataset process

You can download MSR_data_cleaned.csv and Joern by running `get_data.sh`

### Step 1: Clean Code 

Run `scripts/process_dataset.py` to clean dataset and remove abnormal functions, which also get the glove, word2vec and other models that will be used to initialize node embedding by the graph model

### Step 2: Graph Extraction: Generate CPGs with the help of joern

Run `scripts/processJoern.py` to extract .c file and run Joern to get corresponding edges.json and nodes.json

### Step 3: Image Generation

Run `scripts/getImages.py` to check data after Joern and resample the data to produce [the final balanced dataset](https://drive.google.com/file/d/16tm5TU9CUCePFg6wJh2kz71SZylKv8zw/view), and then generate images(i.e.,CPG)

## Baselines 

### UniXcoder

```shell
cd models/cunixcoder

# run UniXcoder model
chmod u+x run.sh
sh run.sh

# save best-f1 UniXcoder code embeddings(function-level)
python main.py --save_unixcoder_embedding

```

### Devign

```shell
cd models/devign
python main.py
```

### Ivdetect

```shell
cd models/ivdetect
python main.py
```

### Reveal

```shell
cd models/reveal/ggnn
python main.py --save_after_ggnn

cd ..
python main.py
```

 
