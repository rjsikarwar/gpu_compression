%Compulsory-Property Table (CPT)
cpt_main(_, _, _, D, 1, ns) :- D < 4.
cpt_main(_, _, _, D, 1, nsv) :- D < 4.
cpt_main(N, _, C, _, 1, dict) :- C > 50, C < 50000.
cpt_main(N, yes, C, _, 1, rle) :- N/C > 4.
cpt_main(N, yes, C, _, 1, bitmap) :- C < 50.
cpt_aux(_, _, _, D, 1, for) :- D < 4.
cpt_aux(_, yes, _, _, 1, delta).
cpt_aux(_, _, _, _, P, sep) :- P > 1.
cpt_aux(_, _, _, _, P, scale):- P > 1.

%Transitional-Property Table (TPT)
tpt_aux(N, S, C, D, P, [for|Y]):- cpt_aux(N, S, C, D, P, for), tpt_main(N, S, N, D, P, Y).
tpt_aux(N, S, C, D, P, [delta|Y]):- cpt_aux(N, S, C, D, P, delta), tpt_main(N, no, C, 3, P, Y).
tpt_aux(N, S, C, D, P, [sep|Y]):- cpt_aux(N, S, C, D, P, sep), ctpt(N, S, C, D, P, sep, Y). 
tpt_aux(N, S, C, D, P, [scale|Y]):- cpt_aux(N, S, C, D, P, scale), tpt_main(N, S, C, D, 1, Y). 
tpt_main(N, S, C, D, P, [X]):- cpt_main(N, S, C, D, P, X).
tpt_main(N, S, C, D, P, [ns|Y]):- cpt_main(N, S, C, D, P, ns), planer(N, S, C, 4, P, Y).
tpt_main(N, S, C, D, P, [dict|Y]):- cpt_main(N, S, C, D, P, dict), planer(N, S, N, 3, P, Y).
tpt_main(N, S, C, D, P, [rle|Y]):- cpt_main(N, S, C, D, P, rle), ctpt(C, yes, C, D, P, rle, Y). 

%Compound TPT (CTPT)
ctpt(_, _, C, D, _, rle, [X,Y]):- planer(C, yes, C, D, 1, X),planer(C, no, C, 3, 1, Y).
ctpt(N, S, C, D, _, sep, [X,Y]):- planer(N, S, C, D, 1, X),planer(N, no, 100, 3, 1, Y).

planer(N, S, C, D, P, X):- tpt_main(N, S, C, D, P, X).
planer(N, S, C, D, P, X):- tpt_aux(N, S, C, D, P, X).
