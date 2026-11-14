"""


MIT License

Copyright (c) 2021 Ben Coveney

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

"""

Game data for the Turtle Pack (the only pack included): pets, foods, perks
(statuses) and per-turn rules.

Do not edit this data here. data/turtle_pack.yaml is the source of truth;
after changing it, run python -m sapai.build_data to rebuild
sapai/game_data.json, which this module loads. See docs/PATCHING.md.

Pack id "StandardPack" is kept as the internal name of the Turtle Pack.
Pets that are only summoned by abilities or perks are kept so those
abilities work, but they are never stocked in the shop, and neither are
foods created by pet abilities.

"""

# %%
import json
import os

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_data.json"),
    encoding="utf-8",
) as _f:
    data = json.load(_f)

### General rules (gold, costs, limits); per-turn rules are in data["turns"]
RULES = data["rules"]
# %%


################################################################################
#### Creating empty fields for pets and foods
################################################################################


def get_fields(d):
    """
    Recursive function to get all possible fields for the input dict

    """
    key_list = []
    for key, value in d.items():
        temp_key_list = [key]

        if type(value) == dict:
            temp_key_list.append(get_fields(value))

        key_list.append(temp_key_list)

    return key_list


def add_dummy_fields(fields, d):
    if d == "none":
        ### Sometimes there are dict or int datatype collisions in the give
        ### json information. This should probably be corrected.
        return

    if type(fields[0]) == str:
        if len(fields) == 1:
            ### Break recursion condition
            d[fields[0]] = "none"
            return
        if type(fields[1]) == list:
            if fields[0] not in d:
                d[fields[0]] = {}

            for temp_fields in fields[1:]:
                add_dummy_fields(temp_fields, d[fields[0]])
    else:
        for temp_list in fields:
            add_dummy_fields(temp_list, d)


dummy_pet = {}
for temp_pet, value in data["pets"].items():
    temp_all_fields = get_fields(value)
    add_dummy_fields(temp_all_fields, dummy_pet)
data["pets"]["pet-none"] = dummy_pet

dummy_foods = {}
for temp_pet, value in data["foods"].items():
    temp_all_fields = get_fields(value)
    add_dummy_fields(temp_all_fields, dummy_foods)
data["foods"]["food-none"] = dummy_foods

# %%
