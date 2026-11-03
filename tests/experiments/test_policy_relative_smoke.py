import numpy as np
import pytest
from scripts.training.policy_relative_smoke import weighted_midpoints, valid_board, level_key


def test_ties_share_success_weighted_interval_not_offset_rank():
    weights,u=weighted_midpoints([7,2,7,4],[1,2,3,2])
    np.testing.assert_allclose(weights,[.125,.25,.375,.25])
    np.testing.assert_allclose(u,[.75,.125,.75,.375])
    assert weights.sum()==1


def test_zero_hit_targets_cannot_receive_labels():
    with pytest.raises(ValueError):
        weighted_midpoints([1,2],[4,0])


def test_board_validation_preserves_static_layout_and_piece_counts():
    board=np.ones((12,12),dtype=int)
    board[[0,-1]]=0
    board[:,[0,-1]]=0
    board[2,2:6]=2
    board[3,2:6]=4
    board[5,5]=5
    assert valid_board(board,board)
    missing=board.copy();missing[3,2]=1
    assert not valid_board(missing,board)
    moved=board.copy();moved[5,5]=1;moved[5,6]=5
    assert valid_board(moved,board)
    assert level_key(np.eye(7)[board])==level_key(np.eye(7)[moved])
