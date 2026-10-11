import json
from pathlib import Path
import pytest
from yue2.protocol import GenerationConfig, Sampling, PROTOCOL_VERSION

@pytest.mark.parametrize('field,value', [('version','future-protocol'),('version',None),('abc',None),('abc',{}),('semantic',[]),('semantic',False)])
def test_invalid_generation_contract_is_rejected_immediately(field, value):
    with pytest.raises((TypeError, ValueError)): GenerationConfig(**{field:value})

def test_dictionary_overrides_still_resolve_sampling_defaults():
    result = GenerationConfig.from_dict({'version':PROTOCOL_VERSION,'abc':{'temperature':.4},'semantic':{'top_k':12}})
    assert isinstance(result.abc, Sampling) and result.abc.temperature == .4 and result.abc.max_tokens == 4096
    assert result.semantic.top_k == 12 and result.semantic.max_tokens == 9000
    assert GenerationConfig.from_dict(result.to_dict()) == result
