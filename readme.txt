Running yourself

0. Clone repo, install dependencies

1. Download data
  https://www.kaggle.com/jessicali9530/stanford-dogs-dataset
  https://www.kaggle.com/ma7555/cat-breeds-dataset

2. Extract data
  Create a direcotry named data in the root of your cloned repo
  Extract the datasets to that direcotry with the following structure:
    data/original_data/cat_breeds (Place cat dataset here)
    data/original_data/dog_breeds (Place dog dataset here)

2. Preprocess data
  ./data.py -c preprocess -i data/original_data -o data/preprocessed
  This will preprocess all data and place the results in preprocessed

3. Run Approach 1 on preprocessed data
  ./data.py -c a1 -i data/preprocessed -o data/a1
  This will process all data and place the results in a1

4. Run Approach 2 on preprocesed data
  ./data.py -c a2 -i data/preprocessed -o data/a2
  This will process all data and place the results in a2

5. Run Approach 3 on preprocessed data
  ./data.py -c a3 -i data/preprocessed -o data/a3
  This will process all data and place the results in a3

From this point, you should be able to run the notebook and produce results
