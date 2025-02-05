# import os

# from hydra import compose
# from hydra import initialize

# import carl.run

# CI_OS_CPU_COUNT = 2

CI_OS_CPU_COUNT = 2

# def test_value_training():
#     with initialize(config_path='./config'):
#         cfg = compose(config_name='test_value_network_transformer')
#         carl.run.run(config=cfg)

    # # @pytest.mark.skipif(os.cpu_count() <= CI_OS_CPU_COUNT, reason='CI is too small')
    # def test_cllp_training_pl():
    #     with initialize(config_path='./config'):
    #         cfg = compose(config_name='test_cllp_network_transformer_pl')
    #         carl.run.run(config=cfg)

    # @pytest.mark.skipif(os.cpu_count() <= CI_OS_CPU_COUNT, reason='CI is too small')
    # def test_cllp_training_hf():
    #     with initialize(config_path='./config'):
    #         cfg = compose(config_name='test_cllp_network_transformer_hf')
    #         carl.run.run(config=cfg)

    # @pytest.mark.skipif(os.cpu_count() <= CI_OS_CPU_COUNT, reason='CI is too small')
    # def test_generator_training():
    #     with initialize(config_path='./config'):
    #         cfg = compose(config_name='test_generator_network_transformer')
    #         carl.run.run(config=cfg)

    # def test_policy_network_training():
    #     with initialize(config_path='./config'):
    #         cfg = compose(config_name='test_policy_network_transformer')
    #         carl.run.run(config=cfg)
