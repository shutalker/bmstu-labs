% table 1
spherical_prop('2.910.01', m6).
spherical_prop('2.910.02', m8).
spherical_prop('3.910.01', m12).
spherical_prop('3.910.02', m16).

% table 2
notched_prop('2.911.01', m6).
notched_prop('2.911.02', m8).
notched_prop('3.911.01', m12).

% table 3
pintle_prop('2.213.01', m6, D) :- D >= 6, D =< 8.
pintle_prop('2.213.04', m8, D) :- D >= 8, D =< 12.
pintle_prop('3.213.06', m12, D) :- D >= 12, D =< 36.

% table 4
threaded_cam('2.913.05', '30_18', m6, 16).
threaded_cam('2.913.06', '45_22', m8, 20).
threaded_cam('2.913.09', '65_30', m12, 38).

% table 5
prismatic_cam('2.913.01', '30_18', 10, Dh, Dv) :-
    (number(Dh), Dh >= 8, Dh =< 12; number(Dv), Dv >= 3, Dv =< 7).

prismatic_cam('2.913.02', '45_22', 12, Dh, Dv) :-
    (number(Dh), Dh >= 8, Dh =< 12; number(Dv), Dv >= 3, Dv =< 7).

prismatic_cam('2.913.07', '65_30', 15, Dh, Dv) :-
    (number(Dh), Dh >= 12, Dh =< 30; number(Dv), Dv >= 8, Dv =< 18).

find_diameter_limits(Bounds, DiameterLimits) :-
    flatten(Bounds, Limits), %TODO: optimise limits searching
    max_list(Limits, MaxLimit),
    min_list(Limits, MinLimit),
    DiameterLimits = [MinLimit, MaxLimit].

get_diameter_limits(blank_cyl_hor, ClauseHead, DiameterLimits) :-
    findall(
        [LowBound, HighBound],
        clause(ClauseHead, ;(','(_, ','(>=(_, LowBound), =<(_, HighBound))), _)),
        Bounds
    ),
    find_diameter_limits(Bounds, DiameterLimits).

get_diameter_limits(blank_cyl_vert, ClauseHead, DiameterLimits) :-
    findall(
        [LowBound, HighBound],
        clause(ClauseHead, ;(_, ','(_, ','(>=(_, LowBound), =<(_, HighBound))))),
        Bounds
    ),
    find_diameter_limits(Bounds, DiameterLimits).

get_diameter_limits(blank_perfor, ClauseHead, DiameterLimits) :-
    findall(
        [LowBound, HighBound],
        clause(ClauseHead, ','(>=(_, LowBound), =<(_, HighBound))),
        Bounds
    ),
    find_diameter_limits(Bounds, DiameterLimits).


% table 6
cam_clamp('2.451.01', '45_30', '30_18', 29).
cam_clamp('2.451.02', '60_45', '45_22', 34).
cam_clamp('3.451.01', '60_45', '45_22', 35).
cam_clamp('3.451.02', '90_60', '65_30', 42).

% table 7
gasket('2.217.01', '45_30', 5).
gasket('2.217.07', '45_30', 3).
gasket('2.217.09', '45_30', 2).
gasket('2.217.10', '45_30', 1).

gasket('3.217.01', '60_45', 5).
gasket('3.217.07', '60_45', 3).
gasket('3.217.09', '60_45', 2).
gasket('3.217.10', '60_45', 1).

gasket('3.107.25', '90_60', 5).
gasket('3.107.27', '90_60', 3).
gasket('3.107.28', '90_60', 2).


% gasket pack rules
gasket_pack(_, PackHeight, _) :- PackHeight < 0, !, fail.
gasket_pack(_, 0, []) :- !.
gasket_pack(TypeSize, PackHeight, [GasketCode | GasketCodeList]) :-
    gasket(GasketCode, TypeSize, Height),
    RemainHeight is PackHeight - Height,
    RemainHeight >= 0,
    gasket_pack(TypeSize, RemainHeight, GasketCodeList), !.

% rules for moveable part
blank(blank_plane_clean, undefined).
blank(blank_plane_rough, undefined).
blank(blank_perfor, D) :- number(D).
blank(blank_cyl_vert, D) :- number(D).
blank(blank_cyl_hor, D) :- number(D).

moveable_part0(blank_plane_clean, _, PropCode, CamCode, CamTypeSize, CamHeight) :-
    spherical_prop(PropCode, PropDm),
    threaded_cam(CamCode, CamTypeSize, PropDm, CamHeight).

moveable_part0(blank_plane_rough, _, PropCode, CamCode, CamTypeSize, CamHeight) :-
    notched_prop(PropCode, PropDm),
    threaded_cam(CamCode, CamTypeSize, PropDm, CamHeight).

moveable_part0(blank_perfor, BlankD, PropCode, CamCode, CamTypeSize, CamHeight) :-
    (
        pintle_prop(PropCode, PropDm, BlankD);
        get_diameter_limits(blank_perfor, pintle_prop(PropCode, PropDm, BlankD), Limits),
	write("Wrong diameter value, it should be in inerval: "), write(Limits), nl, !, fail
    ),
    threaded_cam(CamCode, CamTypeSize, PropDm, CamHeight).

moveable_part0(blank_cyl_vert, BlankD, PropCode, CamCode, CamTypeSize, CamHeight) :-
    PropCode = no_prop,
    (
        prismatic_cam(CamCode, CamTypeSize, CamHeight, _, BlankD);
        get_diameter_limits(blank_cyl_vert, prismatic_cam(CamCode, CamTypeSize, CamHeight, _, BlankD), Limits),
        write("Wrong diameter value, it should be in inerval: "), write(Limits), nl, !, fail
    ).

moveable_part0(blank_cyl_hor, BlankD, PropCode, CamCode, CamTypeSize, CamHeight) :-
    PropCode = no_prop,
    (
        prismatic_cam(CamCode, CamTypeSize, CamHeight, BlankD, _);
        get_diameter_limits(blank_cyl_hor, prismatic_cam(CamCode, CamTypeSize, CamHeight, BlankD, _), Limits),
        write("Wrong diameter value, it should be in inerval: "), write(Limits), nl, !, fail
    ).

moveable_part(BlankType, BlankD, PropCode, CamCode, CamTypeSize, CamHeight) :-
    blank(BlankType, BlankD),
    moveable_part0(BlankType, BlankD, PropCode, CamCode, CamTypeSize, CamHeight).

count([], _ , []) :- !.
count([CSHead | CSTail], Codes, [[CSHead, CodeCounter] | CodesCount]) :-
    findall(CSHead, member(CSHead, Codes), TempList),
    length(TempList, CodeCounter),
    count(CSTail, Codes, CodesCount).

count(Codes, CodesCount) :-
    is_list(Codes),
    list_to_set(Codes, CodeSet),
    count(CodeSet, Codes, CodesCount).

pretty_print([], _) :- !.
pretty_print(CodeList, PartName) :-
    count(CodeList, CodesCount), !,
    write(PartName), write(": "), write(CodesCount), nl.

% rules for device
assembly_device(BlankType, BlankD, Height) :-
    blank(BlankType, BlankD), !,
    moveable_part(BlankType, BlankD, PropCode, CamCode, CamTypeSize, CamHeight),
    cam_clamp(ClampCode, ClampTypeSize, CamTypeSize, ClampHeight),
    PackHeight is Height - CamHeight - ClampHeight - 30,
    gasket_pack(ClampTypeSize, PackHeight, Gaskets),
    pretty_print(Gaskets, "gaskets"),
    pretty_print([ClampCode], "clamps"),
    pretty_print([CamCode], "cams"),
    pretty_print([PropCode], "props").
















