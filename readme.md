# Project Information
I have been working as a data engineer for many years, early in my career I was a data mining engineer.
As of today, AI is really everywhere, I want to do some small experiments on LLM to learn some fun tech.

Lets do it together.

## Update 03-23-2025

### I just found out, it is pretty hard to understand the training and prediction process. I want to set targets to finish this.

* add more documents to explain training and prediction on LLM
* add more code examples to show how training works in a smaller code piece

## Update 03-21-2025

### I want to train a simple model with several sentences to see how it works

* ~~change input from file to folder~~
* ~~add mlflow and optuna~~

lets have fun on this.


### how to run this tiny project

* ``` pip install -r requirements.txt ```

* ``` python train.py```

```

[I 2025-03-21 23:41:10,912] A new study created in memory with name: no-name-5e5186c0-7af1-4d54-99f1-b15f7fb752ed
[I 2025-03-21 23:41:11,807] Trial 0 finished with value: 0.6095694303512573 and parameters: {'n_head': 2, 'base_embed': 29, 'n_layer': 3, 'block_size': 5, 'dropout': 0.36080348341790525, 'batch_size': 2}. Best is trial 0 with value: 0.6095694303512573.
[I 2025-03-21 23:41:12,024] Trial 1 finished with value: 1.0326344966888428 and parameters: {'n_head': 2, 'base_embed': 19, 'n_layer': 1, 'block_size': 4, 'dropout': 0.19908629564116564, 'batch_size': 4}. Best is trial 0 with value: 0.6095694303512573.
[I 2025-03-21 23:41:12,246] Trial 2 finished with value: 0.38251280784606934 and parameters: {'n_head': 4, 'base_embed': 16, 'n_layer': 3, 'block_size': 4, 'dropout': 0.3244054171136306, 'batch_size': 2}. Best is trial 2 with value: 0.38251280784606934.
[I 2025-03-21 23:41:12,401] Trial 3 finished with value: 0.25447872281074524 and parameters: {'n_head': 4, 'base_embed': 27, 'n_layer': 3, 'block_size': 5, 'dropout': 0.3130428244814018, 'batch_size': 1}. Best is trial 3 with value: 0.25447872281074524.
[I 2025-03-21 23:41:12,578] Trial 4 finished with value: 0.6659662127494812 and parameters: {'n_head': 3, 'base_embed': 31, 'n_layer': 3, 'block_size': 4, 'dropout': 0.41810735684031747, 'batch_size': 1}. Best is trial 3 with value: 0.25447872281074524.
[I 2025-03-21 23:41:12,655] Trial 5 finished with value: 2.4010732173919678 and parameters: {'n_head': 2, 'base_embed': 32, 'n_layer': 1, 'block_size': 3, 'dropout': 0.4967368063152342, 'batch_size': 1}. Best is trial 3 with value: 0.25447872281074524.
[I 2025-03-21 23:41:12,735] Trial 6 finished with value: 2.396498918533325 and parameters: {'n_head': 1, 'base_embed': 18, 'n_layer': 1, 'block_size': 3, 'dropout': 0.1174563772175183, 'batch_size': 1}. Best is trial 3 with value: 0.25447872281074524.
[I 2025-03-21 23:41:12,803] Trial 7 finished with value: 1.4618948698043823 and parameters: {'n_head': 3, 'base_embed': 17, 'n_layer': 1, 'block_size': 3, 'dropout': 0.47522119449291444, 'batch_size': 4}. Best is trial 3 with value: 0.25447872281074524.
[I 2025-03-21 23:41:12,968] Trial 8 finished with value: 0.2234538495540619 and parameters: {'n_head': 3, 'base_embed': 26, 'n_layer': 3, 'block_size': 4, 'dropout': 0.24987475928350555, 'batch_size': 4}. Best is trial 8 with value: 0.2234538495540619.
best parameters: {'n_head': 3, 'base_embed': 26, 'n_layer': 3, 'block_size': 4, 'dropout': 0.24987475928350555, 'batch_size': 4}
best loss: 0.2234538495540619
[I 2025-03-21 23:41:13,051] Trial 9 finished with value: 0.9419061541557312 and parameters: {'n_head': 2, 'base_embed': 25, 'n_layer': 1, 'block_size': 4, 'dropout': 0.18404792476163287, 'batch_size': 2}. Best is trial 8 with value: 0.2234538495540619.
Process finished with exit code 0
```


