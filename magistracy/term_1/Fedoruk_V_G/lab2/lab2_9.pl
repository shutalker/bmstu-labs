% parse input string into list of symbols
% hack: place right bracket as the last token to process right part of equation
token_parser([], [')']).

% skip whitespaces
token_parser([SymCode | InputSymbols], Tokens) :-
    char_type(SymCode, white),
    token_parser(InputSymbols, Tokens).

%parse upper letters (Z, Y, X, ...) as variables
token_parser([SymCode | InputSymbols], [_ | Tokens]) :-
    char_type(SymCode, upper),
    token_parser(InputSymbols, Tokens).

% parse digits
token_parser([SymCode | InputSymbols], [ParsedToken | Tokens]) :-
    char_type(SymCode, digit),
    ParsedToken is SymCode - 48,
    token_parser(InputSymbols, Tokens).

% parse parentheses and operators
token_parser([40 | InputSymbols], ['(' | Tokens]) :- token_parser(InputSymbols, Tokens).
token_parser([41 | InputSymbols], [')' | Tokens]) :- token_parser(InputSymbols, Tokens).
token_parser([42 | InputSymbols], ['*' | Tokens]) :- token_parser(InputSymbols, Tokens).
token_parser([43 | InputSymbols], ['+' | Tokens]) :- token_parser(InputSymbols, Tokens).
token_parser([45 | InputSymbols], ['-' | Tokens]) :- token_parser(InputSymbols, Tokens).
token_parser([47 | InputSymbols], ['/' | Tokens]) :- token_parser(InputSymbols, Tokens).

% hack: replace '=' symbol with "- ( ... )" sequence
token_parser([61 | InputSymbols], ['-', '(' | Tokens]) :- token_parser(InputSymbols, Tokens).

% parse list of symbols into list of lexems
lexem_parser([], []).
lexem_parser([Token], [Token]).

% parse variables or parentheses or operators
lexem_parser([Token1, Token2 | Tokens], [Token1 | Lexems]) :-
    (var(Token1); atom(Token1)),
    lexem_parser([Token2 | Tokens], Lexems).

% parse single-digit number
lexem_parser([Token1, Token2 | Tokens], [Token1 | Lexems]) :-
    integer(Token1),
    (var(Token2); atom(Token2)),
    lexem_parser([Token2 | Tokens], Lexems).

% parse number from digits
lexem_parser([Token1, Token2 | Tokens], Lexems) :-
    integer(Token1),
    integer(Token2),
    Number is Token1 * 10 + Token2,
    lexem_parser([Number | Tokens], Lexems).

% syntax Definite Clause Grammar parser
ex(E) --> eterm(E).
ex([S, E1, E2]) --> sterm(E1), sn(S), eterm(E2).

sterm(E) --> eterm(E).
sterm([S, E1, E2]) --> eterm(E1), sn(S), eterm(E2).
sterm([S2, [S1, E1, E2], E3]) --> eterm(E1), sn(S1), sterm(E2), sn(S2), eterm(E3).

eterm(E) --> fct(E).
eterm([S2, [S1, E1, E2], E3]) --> fct(E1), sn2(S1), eterm(E2), sn2(S2), fct(E3).
eterm([S, E1, E2]) --> fct(E1), sn2(S), fct(E2).

sn2(X) --> [X], {var(X)}.
sn2(*) --> [*].
sn2(/) --> [/].

fct(E) --> number(E).
fct(E) --> lb, ex(E), rb.
fct(E) --> sn(E), fct(E).

number(X) --> [X], {integer(X)}.

lb --> ['('].
rb --> [')'].

sg(X) --> [X], {var(X)}.
sg(+) --> [+].
sg(-) --> [-].
sn(E) --> sg(E).

syntax_parser(Lexems, Expression) :- ex(Expression, Lexems, []).

calc([S, A1, A2], Nr) :-
    calc(A1, N1),
    calc(A2, N2),
    calc1(S, N1, N2, Nr).

calc(A1, A1) :- A1 \= [_ | _].
calc1(*, N1, N2, Nr) :- Nr is N1 * N2.
calc1(/, N1, N2, Nr) :- Nr is N1 / N2.
calc1(+, N1, N2, Nr) :- Nr is N1 + N2.
calc1(-, N1, N2, Nr) :- Nr is N1 - N2.

lab2 :-
    read(Input),
    string_codes(Input, InputSymCodes),
    token_parser(InputSymCodes, Tokens),
    write("Tokens = "), write(Tokens), nl,
    lexem_parser(Tokens, Lexems),
    write("Lexems = "), write(Lexems), nl,
    syntax_parser(Lexems, Expression),
    write("Source Expression = "), write(Expression), nl, !,
    calc(Expression, Result),
    Result = 0,
    write("Result Expression = "), write(Expression), nl.

















