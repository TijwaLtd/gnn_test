This is the general flow of files to run
(Data is provided under DFT_DATA , all you need to run the GNN MODEL is the OUTCAR and the POSCAR file from each folder)
(Requirements are also provided)

## To run the model
1. Run bulk_export.py to make sure that each file has a corresponding graph.pt which should be done , but if it not done please run it
2. Run generate_index.py to make the index that the model will use to run it (Index folder will have all the references IF needed)
3. Run e3nn_gnn_model.py to run the model 

## To test the model 
1. There are some model files that i linked as well , but if you wish to run it on your own , the model will generate some model.pth files , once thats done
2. open Test files folder 
3. Use Test_model.py to run it to see how the forces are doing , since they are my primary focus as of now (MAKE SURE THE PARAMETERS OF MODEL ARE SAME)
4. Run test_parity.py or test_parity2.py to see how the graph is made for forces, there should not be a line at 0 