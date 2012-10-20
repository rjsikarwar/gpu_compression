%Compressed-Data-Property Table (CDP Table)
cdpt(unsorted, sortable, multivalue, unseparated, ns).
cdpt(unsorted, unsortable, multivalue, separated, nsv).
cdpt(unsorted, sortable, multivalue, unseparated, dict).
cdpt(unsorted, sortable, multivalue, separated, rle).
cdpt(unsorted, sortable, onevalue, unseparated, bitmap).
cdpt(unsorted, sortable, multivalue, unseparated, for).
cdpt(unsorted, unsortable, multivalue, unseparated, delta).
cdpt(unsorted, unsortable, multivalue, separated, sep).

%Query-applicable Table for soring
qat(_, _, _, separated, sorting, no).
qat(unsorted, unsortable, _, unseparated, sorting, no).
qat(_, sortable, _, unseparated, sorting, yes).

%Query-applicable Table for other operations
%...

optimizer([], _, []).
optimizer([X|_], OP, []) :- cdpt(S, ST, O, SP, X), qat(S, ST, O, SP, OP, R), R=no.
optimizer([X|Y], OP, [X|Z]) :- cdpt(S, ST, O, SP, X), qat(S, ST, O, SP, OP, R), R=yes, optimizer(Y, OP, Z).
