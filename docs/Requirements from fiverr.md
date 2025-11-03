The base readme.md is just an old md that explains what it does , theres no need to follow any requirements written on there since those are already fixed. It just gives the context for the files. 

As we discussed , the issue im having is that my model is not predicting the forces around 0 properly, this is an issue that happens to many GNN models when they try to predict the energies and forces. The way it works is that the model trains on non equilibrium data which is usually not containing many 0 forces , so when presented with equilibrium data which has more 0 forces it fails to properly capture those and hence causes issues

I have tried the following fixes;
1. Since I have around 2200 non-equilibrium structures (with non-zero forces), I added 100 equilibrium structures to the dataset. This helps the model better learn the distribution of near-zero forces but the change is VERY minimal

2. I also introduced an epsilon policy — a small constant 𝜀
ε added during training to stabilize gradients and prevent the network from over-penalizing small-magnitude forces. This policy ensures that near-zero forces are not lost in numerical noise and encourages smoother convergence around the equilibrium regime. More on this can be found on my e3nn_gnn_model under compute forces where you will be able to see it

So, 
Requirements from you guys would be to 
1. Fix the 0 forces errors so the model is more accurate or reduce it AS MUCH as possible where its only a small bump. 
2. Try to handle energies if you can , if not its TOTALLY FINE
3. Clean the file structure and make sure everything works after the files are properly structured.

4. (OPTIONAL) Try to see if you can research on kalman filter but only implement it if it improves the model if not ignore it.