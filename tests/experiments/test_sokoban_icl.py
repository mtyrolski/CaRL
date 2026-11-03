import json
import numpy as np
import pytest
from scripts.sokoban_icl import board_rows,parse_board,build_messages,make_requests


def example():
    return dict(state=['############']*12,goal=['#----------#']*12,target=['#...........']*12,u=.75,d=6,
                context=1,private_success_rate=.9)


def test_board_roundtrip_and_malformed_output():
    board=np.arange(144).reshape(12,12)%7
    np.testing.assert_array_equal(parse_board({'rows':board_rows(board)}),board)
    with pytest.raises(ValueError):parse_board({'rows':['#']*12})
    with pytest.raises(ValueError):parse_board({'rows':['?'*12]*12})


def test_query_does_not_leak_target_other_coordinate_or_calibration():
    case=example();messages=build_messages(case,[], 'relative',20)
    query=json.loads(messages[-1]['content'])
    assert set(query)=={'state','goal','u'}
    assert 'target' not in query and 'd' not in query
    assert set(json.loads(build_messages(case,[],'absolute',20)[-1]['content']))=={'state','goal','d'}
    assert set(json.loads(build_messages(case,[],'none',20)[-1]['content']))=={'state','goal'}


def test_request_budget_checked_before_network():
    config=dict(conditions=['relative','absolute'],execution_cap=20,model='example',max_output_tokens=100,
                max_requests=1,input_usd_per_million=1,output_usd_per_million=1,max_estimated_cost_usd=1)
    with pytest.raises(ValueError,match='Request count'):make_requests(config,{'queries':[example()],'demonstrations':[]})
