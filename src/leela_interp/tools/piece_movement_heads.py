good_knight_heads = [
    (0, 17),
    (1, 19),
    (3, 14),
    (3, 15),
    (4, 3),
    (5, 3),
    (5, 13),
    (8, 5),
    (9, 5),
    (14, 22),
]

noisy_knight_heads = [
    (2, 15),
    (6, 0),
    (6, 10),
    (7, 11),
    (8, 19),
    (9, 4),
    (10, 12),
    (10, 22),
    (11, 17),
    (12, 19),
    (12, 22),
    (13, 0),
]

good_bishop_heads = [
    (1, 4),
    (1, 16),
    (3, 0),
    (3, 8),
    (4, 10),
    (4, 15),
    (4, 17),
    (5, 17),
    (5, 19),
    (6, 3),
    (6, 20),
    (7, 5),
    (7, 18),
    (8, 1),
    (9, 1),
    (10, 0),
    (11, 5),
    (11, 7),
    (11, 16),
    (12, 20),
    (13, 14),
    (14, 20),
]

noisy_bishop_heads = [
    (0, 10),
    (4, 14),
    (6, 4),
    (10, 8),
    (13, 18),
]

good_rook_heads = [
    (0, 15),
    (1, 10),
    (1, 20),
    (2, 7),
    (2, 12),
    (3, 16),
    (4, 18),
    (5, 10),
    (6, 9),
    (7, 13),
    (7, 14),
    (7, 19),
    (7, 22),
    (9, 3),
    (9, 12),
    (10, 6),
    (10, 11),
    (10, 23),
    (14, 16),
]

noisy_rook_heads = [
    (3, 19),
    (4, 1),
    (4, 22),
    (8, 14),
    (11, 12),
    (11, 19),
    (11, 22),
    (12, 9),
    (13, 4),
    (14, 2),
]

all_heads = (
    good_knight_heads
    + noisy_knight_heads
    + good_bishop_heads
    + noisy_bishop_heads
    + good_rook_heads
    + noisy_rook_heads
)
# no duplicates
assert len(set(all_heads)) == len(all_heads)


knight_heads = good_knight_heads + noisy_knight_heads
bishop_heads = good_bishop_heads + noisy_bishop_heads
rook_heads = good_rook_heads + noisy_rook_heads
