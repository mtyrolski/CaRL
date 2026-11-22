"""Prepare matched LoRA generators from measured policy-relative intents.

This module never trains the controller or calls its API.  A calibration file
contains all candidates in each context, including failures, and one complete
set of measured attempts per candidate.  The supplied ``u`` and ``weight``
are checked against those attempts before successful candidates are used.

Every row requires ``context_id`` and ``policy_measurements`` with
``controller_hash``, ``action_cap``, ``expected_candidates``, ``complete``,
``attempts``, ``success_fraction``, ``mean_success_actions``, ``u`` and
``weight``.  Each attempt has ``status`` and ``primitive_actions_attempted``.
The producer must separately freeze the planned candidate IDs: a declared
candidate count alone cannot prove that the selected pool was not changed.

Generator inputs are restricted to the current observation, task and either
an explicitly requested scale or the missing-scale marker.  Evaluation uses
separate test contexts without reference intents or future observations."""

# Same-name aliases declare the compatibility facade's re-exports to type checkers.
# pylint: disable=useless-import-alias,unused-import
from carl.icl.boards import canonical_layout_id as canonical_layout_id
from carl.icl.boards import from_rows as from_rows
from carl.icl.intent_data import validate_prediction as validate_prediction
from carl.icl.intent_generator_experiment import SYSTEM as SYSTEM
from carl.icl.intent_generator_experiment import DataSplit as DataSplit
from carl.icl.intent_generator_experiment import atomic_json as atomic_json
from carl.icl.intent_generator_experiment import file_sha256 as file_sha256
from carl.icl.intent_generator_experiment import generate as generate
from carl.icl.intent_generator_experiment import initialise as initialise
from carl.icl.intent_generator_experiment import memory as memory
from carl.icl.intent_generator_experiment import prompt_messages as prompt_messages
from carl.icl.intent_generator_experiment import read_records as read_records
from carl.icl.intent_generator_experiment import sync_gpu as sync_gpu
from carl.icl.intent_generator_experiment import target_text as target_text
from carl.icl.intent_generator_experiment import tensor_example as tensor_example
from carl.icl.intent_representation_training import completion_status as completion_status
from carl.icl.intent_representation_training import expired as expired
from carl.icl.intent_representation_training import save_adapter as save_adapter
from carl.icl.intents.representation.normalization import remove_one_outer_fence as remove_one_outer_fence
from carl.icl.policy_relative.checks import _attempts as _attempts
from carl.icl.policy_relative.checks import _checked_context as _checked_context
from carl.icl.policy_relative.checks import _digest_string as _digest_string
from carl.icl.policy_relative.checks import _equal as _equal
from carl.icl.policy_relative.checks import _identity as _identity
from carl.icl.policy_relative.checks import _intent_key as _intent_key
from carl.icl.policy_relative.checks import _number as _number
from carl.icl.policy_relative.checks import _positive_int as _positive_int
from carl.icl.policy_relative.cli import main as main
from carl.icl.policy_relative.configuration import _candidate_lengths as _candidate_lengths
from carl.icl.policy_relative.configuration import _configuration as _configuration
from carl.icl.policy_relative.configuration import _model_revision as _model_revision
from carl.icl.policy_relative.configuration import require_scale_information as require_scale_information
from carl.icl.policy_relative.data import load_demo_data as load_demo_data
from carl.icl.policy_relative.data import load_fit_data as load_fit_data
from carl.icl.policy_relative.data import load_study_data as load_study_data
from carl.icl.policy_relative.evaluation import _strict_json as _strict_json
from carl.icl.policy_relative.evaluation import assess_output as assess_output
from carl.icl.policy_relative.evaluation import evaluate as evaluate
from carl.icl.policy_relative.evaluation import test_contexts as test_contexts
from carl.icl.policy_relative.evaluation import verify_adapter as verify_adapter
from carl.icl.policy_relative.prompts import balanced_samples as balanced_samples
from carl.icl.policy_relative.prompts import candidate_scale as candidate_scale
from carl.icl.policy_relative.prompts import encode_example as encode_example
from carl.icl.policy_relative.prompts import encode_target as encode_target
from carl.icl.policy_relative.prompts import generator_messages as generator_messages
from carl.icl.policy_relative.prompts import study_samples as study_samples
from carl.icl.policy_relative.prompts import training_prompts as training_prompts
from carl.icl.policy_relative.resource import resource_check as resource_check
from carl.icl.policy_relative.resource import resource_encodings as resource_encodings
from carl.icl.policy_relative.resource import resource_records as resource_records
from carl.icl.policy_relative.training import _loss as _loss
from carl.icl.policy_relative.training import benchmark as benchmark
from carl.icl.policy_relative.training import train as train
from carl.icl.policy_relative.training import validation_loss as validation_loss
from carl.icl.policy_relative.types import MISSING_SCALE as MISSING_SCALE
from carl.icl.policy_relative.types import SCALE_INSTRUCTION as SCALE_INSTRUCTION
from carl.icl.policy_relative.types import Arm as Arm
from carl.icl.policy_relative.types import AttemptStatus as AttemptStatus
from carl.icl.policy_relative.types import Candidate as Candidate
from carl.icl.policy_relative.types import Context as Context
from carl.icl.policy_relative.types import FitData as FitData
from carl.icl.policy_relative.types import Mode as Mode
from carl.icl.policy_relative.types import PredictionStatus as PredictionStatus
# pylint: enable=useless-import-alias,unused-import

if __name__ == "__main__":
    main()
