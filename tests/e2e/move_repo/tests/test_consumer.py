from pkg.consumer import answer


def test_answer():
    assert answer() == 42
