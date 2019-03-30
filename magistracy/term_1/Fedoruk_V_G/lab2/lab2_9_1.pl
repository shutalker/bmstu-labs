% arithmetic equation syntax parser
equation([Lp, Rp]) --> leftpart(Lp), eq, rightpart(Rp).

leftpart(Lp) --> expression(Lp).
rightpart(Rp) --> expression(Rp).

expression(E) --> term(E).
expression([T, As, E]) --> term(T), adsign(As), expression(E).

eq --> ['='].

term(T) --> multiplexer(T).
term([M, Ms, T]) --> multiplexer(M), mulsign(Ms), term(T).

multiplexer(E) --> leftbracket, expression(E), rightbracket.
multiplexer(N) --> [N], {integer(N)}.

leftbracket --> ['('].
rightbracket --> [')'].

adsign(+) --> ['+'].
adsign(-) --> ['-'].
adsign(As) --> [As], {var(As)}.

mulsign(*) --> ['*'].
mulsign(/) --> ['/'].
mulsign(Ms) --> [Ms] , {var(Ms)}.


%ex -->  sterm.
%ex -->  sterm, adsign,ex.
%sterm --> adsign, term.
%sterm --> term.
%term --> factor.
%term --> factor,musign,term1.
%term1 --> term.
%term1 --> term, musign, term1.
%factor --> [N],{number(N)}.
%factor --> lb,ex,rb.
%adsign --> ['+']. adsign --> ['-'].
%musign --> ['*']. musign --> ['/'].
%lb --> ['(']. rb --> [')'].

%find_operators(Lexems) :-
%    equation(Lexems, Expression),
%    write(Expression), nl.



















