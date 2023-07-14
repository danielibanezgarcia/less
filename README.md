# LeSS

### A Computationally-Light Lexical Simplifier for Spanish

----

# Simplification

## Setup

### 1. Create a Conda environment (requires [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html))

   ```buildoutcfg
   conda create -n less python=3.9 -y
   conda activate less
   ```

### 2. Clone and install the LeSS repository

   ```buildoutcfg
   git clone https://github.com/danielibanezgarcia/less.git
   cd less
   python -m pip install -r requirements.txt
   python -m pip install -e . 
   python -m spacy download es_core_news_lg
   ```
   
### 3. Install Freeling

   [Here](https://freeling-user-manual.readthedocs.io/en/v4.2/) the Freeling user manual.

### 4. Install Morphogen Generator for Spanish

   Requirements:
   - jpype
   - java 1.8 jdk
   
   
   1. install java 8: `sudo apt-get install openjdk-8-jdk` 

   2. get the jpype code: https://github.com/jpype-project/jpype/

   3. compile jpype1 with java 1.8: `JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/ python setup.py install`

### 5. Download models

- Download [GloVe 300 file](https://drive.google.com/file/d/1WNChGQ7ANhHa48b5qfxm8FwvAle-6mCv/view?usp=sharing) under ```data/embeddings/cc.es.300.vec```
- Download [ngrams file](https://drive.google.com/file/d/1mtL49UFbEQ2cDlFz0XK0GbI1gHa24dgo/view?usp=sharing) under ```data/ngrams/es-2-grams.pkl```

## How to Use It

The expected input file format is a Mechanical Turk Lexical Simplification Data File, as proposed by Horn, Manduca and Kauchak (2014) in [Learning a Lexical Simplifier Using Wikipedia](https://aclanthology.org/P14-2075).

Example:
```
Spanish Sentence	Word	Mturk labels
), Fue un pianista y compositor folclórico, y figura fundamental en lo que a interpretación pianística del folclore argentino se refiere.	folclórico	popular	tradicional	local		de folclore	de musica folk	costumbrista	pintoresco	típico 	tradicional	tradicional	pintoresco	típico	de folclor	tradicional	tradicional	tradicional	tradicional	típico	costumbrista	tradicional	tradicional	tradicional	pintoresco	de folclore
A comienzos de la década de 1980, se trasladó a Los Ángeles, en California, donde comenzó a labrarse una reputación con sus actuaciones, tanto eléctricas como acústicas.	labrarse	construirse	trabajar	ganarse 	formarse	construirse	cultivar	formarse	formarse	formarse 	formarse	construirse	construirse	prepararse	hacerse	trabajarse	ganarse	forjarse	hacerce	crearse	hacerse	hacerse	formarse	crearse	forjarse	hacerse
A lo largo de sus más de veinte años de experiencia en el medio, ha presentado todo tipo de programas, no sólo informativos, sino también divulgativos, de entrevistas, tertulias e incluso concursos.	tertulias	conversaciones	debates	reunion 	debates	charlas	reunión	conversación	reuniones	charlas	charlas	reuniones	charlas	debates	conversaciones	charlas	fiestas	reunión	reuniones	charlas	reuniones 	reuniones	reuniones	debates	coloquios	charlas
...
```

To obtain candidates, just run the horn_simplifier: 

```python
from config import horn_simplifier

input_file_path = 'data/horn/lex.mturk.es.txt'
candidates_file_path = 'data/horn/results/less_candidates.txt'
horn_simplifier.simplify(input_file_path, candidates_file_path)

```

This will produce a file like:

```
folclórico	folklórico	autóctono	dancístico	danzario	folkórico	folcklórico
forjar	consolidar	establecer	formar	buscar	ganar	triunfar	prosperar	cimentar	labrarse	edificar	labrar	labrar	asentar	afianzar	sembrar	medrar	profesionalizar	roturar	arar	arar	afincar	cincelar	cualificar	curtir	bruñir	pulimentar
reuniones	charlas	charlas	charlas	conversaciones	tertulias	conferencias	discusiones	veladas	sobremesas	tabernas	sabatinas	juergas	café-tertulias	contertulias	comidillas	cuchipandas	reboticas	contertulianas
provocado	llevado	producido	propiciado	favorecido	originado	permitido	generado	ocasionado	posibilitado	fomentado	contribuido	impedido	facilitado	promovido	conllevado	desembocado	impulsado	alentado	desencadenado	incentivado	procurado	estimulado	coadyuvado	redundado	propendido
...
```
