[X] Elo: Precompute number of total maximum tasks for the progress bar (must have only one progress bar)
[X] Elo/csv: let matches count be integers
[X] Json: Format validation. Raise error
[X] Remove constructors and the need to write one instruction per property to instanciate an object
[x] Implement Elo draws
[X] Ensure that a single ctrl-c cancels tasks
[] Write end-to-end tests
    [] Setup http cache with one production repository and one test repo
    [X] Add --seed option. Used only in Elo stage to make batch generation deterministic. Caution : this is not about seed/_seed passed to the llm.
    [X] Setup the base code to run an end to end test and its environment, meaning from cli and a json file
    [] Write relevant cases. Ignore testing the console input/output.
[] Implement an alternative way to constrain responses for model providers that do not support json schema like OpenAI
    [X] With prompt engineering and output correction
    [] With function calling
[X] Make csv output path relative to folder of input
[] Elo: Add optional feature to weight victories based on model confidence using response log probabilities
[x] Elo: rename matches_per_entity to a name that indicate it's a batch count and amend this change into commit 0472297