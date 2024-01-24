The Pytorch code of the paper "[Document-level Relation Extraction with Entity Type Constraints](./DocRE-TCGL.pdf)", under review in ``Information Processing and Management`` journal.


## requirements

+ pytorch = 1.7.1
+ cuda = 10.2.89
+ cudnn = 7.6.5
+ transformers = 3.4.0

## How to run?

```
bash ./scripts/docred/run_bert.sh
``` 
or
```
bash ./scripts/dwie/dwie_run_bert.sh
```

**Note**: 

The "``--load_path``" option in the script "``*.sh``" controls whether it is in the training phase or the testing phase.
