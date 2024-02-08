# ACE-2005 Translation and Annotation Alignment Pipeline

This pipeline translates the  [ACE 2005 corpus](https://catalog.ldc.upenn.edu/LDC2006T06)  into Portuguese and aligns the translated annotations with the corresponding text.
## Overview


This repository contains a Python translation and annotation alignment pipeline that is designed to translate the ACE 2005 dataset into Portuguese using machine translation and then align the translated annotations with the corresponding text. The pipeline was developed for automatic translation of ACE-2005 into Portuguese but can also be adapted for other languages with little effort.

It is composed of two main components: 

    - Text translation: We used DeepL Translator and Google Translator to translate ACE-2005 texts and annotations into European and Brazilian Portuguese. 
    - Annotations alignments: We developed an annotation alignment pipeline that aligns the translated annotations within the translated text.


## Prerequisites
1.  Prepare ACE 2005 dataset

    Download: (https://catalog.ldc.upenn.edu/LDC2006T06). Note that ACE 2005 dataset is not free.)

2. ACE-2005 Pre-processing
    We adopted a commonly used ACE-2005 pre-processing that can be found in [this repository](https://github.com/nlpcl-lab/ace2005-preprocessing). 

3. Install the packages.

    ```bash
    pip install -r requirements.txt
    ```

## Annotation Alignment Modules
Our pipeline is composed of a total of five annotation alignment components:

    - Lemmatization
    - Multiple word translation
    - Synonyms
    - BERT-based word aligner
    - Fuzzy Match (Gestalt Patter Matching and Levenstein distance)

The pipeline operates sequentially, meaning that annotations aligned by earlier methods are not addressed by subsequent pipeline elements. According to our experiments, the list above corresponds to the best order sequence.


## Usage



3. **Translate ACE-2005 to Portuguese**
    By default we use Google Translate for the translation process. An API key is need in order to use DeepL Translator.
    ```bash
    Usage: python3 translation.py <input_file> <output_dir>
    Example: python src/translate.py data/sample_en.json data/sample_pt.json
    ```


4. **Run the Annotation Alignment Pipeline**

    To align the translated annotations, the alignment pipeline can be executed with the following command:

    ```bash
    Usage: python3 pipeline.py <input_file> <output_dir>
    Example: python src/translate.py data/sample_pt.json data/sample_pt_aligned.json
    ```

    The pipeline can be configured in the config.yaml file. Users can change the order of aligners and must indicate the path for the alignment resources for each component, such as multiple translations of annotations, previously calculated lemmas, synonyms, etc. All of these resources are already pre-calculated for the Portuguese language in the resources folder. Additionally, the input and output files can be configured in the config.yaml file as well.

## Evaluation


To measure the effectiveness of the alignment pipeline, manual alignments were conducted on the entire ACE-2005-PT test set, which includes 1,310 annotations (triggers and arguments). These alignments were performed by a linguist expert to ensure high-quality annotations, following the same annotation guidelines of the original ACE-2005 corpus \cite{aceguidelines}.

The evaluation results are presented in Table 1:

<p>
    <img src="./img/eval_by_comp.png" alt="Results" width="500"/>
    <br>
    <em>Table 1: Evaluation Results by pipeline component</em>
</p>





## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).


