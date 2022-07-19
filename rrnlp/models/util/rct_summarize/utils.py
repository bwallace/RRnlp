
from collections import Counter
from itertools import groupby
from torch import nn 

def multimode(l):
   freqs = groupby(Counter(l).most_common(), lambda x:x[1])
   return [val for val,count in next(freqs)[1]]



def load_vanilla_layers(saved_layers, model):
    model_updated_state_dict = model.state_dict()

    for layer_name, layer_params in model.state_dict().items():
        saved_layer_name = 'model.'+layer_name
        if saved_layer_name in saved_layers:
            model_updated_state_dict[layer_name] = saved_layers[saved_layer_name]
        else:
            print(saved_layer_name)
        
    return model_updated_state_dict


def load_multilm_layers(saved_layers, model):
    
    model_updated_state_dict = model.state_dict()

    for layer_name, layer_params in model.state_dict().items():
        saved_layer_name = 'model.'+layer_name

        if saved_layer_name  in saved_layers.keys():
            model_updated_state_dict[layer_name] = saved_layers[saved_layer_name]
            #print('FOUND', saved_layer_name)
        else:
            if 'decoder' in layer_name:
                decoder_key = layer_name.split('.')
                shared_decoder_key =  ['decoder'] + decoder_key[1:]
                shared_decoder_key = '.'.join(shared_decoder_key)
                saved_layer_name = 'model.'+shared_decoder_key
                model_updated_state_dict[layer_name]  = saved_layers[saved_layer_name]
                print('SHARED', layer_name, saved_layer_name)
            else:
                print(layer_name)
    return model_updated_state_dict

def _tie_decoder_weights(decoder1: nn.Module, decoder2: nn.Module, module_name: str):
    def tie_decoder_recursively(
            decoder1_pointer: nn.Module,
            decoder2_pointer: nn.Module,
            module_name: str,
            depth=0,
    ):
        assert isinstance(decoder1_pointer, nn.Module) and isinstance(
            decoder2_pointer, nn.Module
        ), f"{decoder1_pointer} and {decoder2_pointer} have to be of type nn.Module"
        if hasattr(decoder1_pointer, "weight"):
            assert hasattr(decoder2_pointer, "weight")
            decoder1_pointer.weight = decoder2_pointer.weight
            if hasattr(decoder1_pointer, "bias"):
                assert hasattr(decoder2_pointer, "bias")
                decoder1_pointer.bias = decoder2_pointer.bias
            return

        decoder1_modules = decoder1_pointer._modules
        decoder2_modules = decoder2_pointer._modules
        if len(decoder2_modules) > 0:
            assert (
                    len(decoder1_modules) > 0
            ), f"Decoder modules do not match"

            all_decoder_weights = set([module_name + "/" + sub_name for sub_name in decoder1_modules.keys()])
            for name, module in decoder2_modules.items():
                tie_decoder_recursively(
                    decoder1_modules[name],
                    decoder2_modules[name],
                    module_name + "/" + name,
                    depth=depth + 1,
                )
                all_decoder_weights.remove(module_name + "/" + name)

            assert len(all_decoder_weights) == 0, 'There are some extra parameters in one of the decoders'

    # tie weights recursively
    tie_decoder_recursively(decoder1, decoder2, module_name)
