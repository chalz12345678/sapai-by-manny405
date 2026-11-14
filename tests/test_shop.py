# %%
import unittest

from sapai import Player, Team
from sapai.shop import *


class TestShop(unittest.TestCase):
    def test_shop_slot_pet(self):
        slot = ShopSlot("pet")
        slot.obj = Pet("ant")
        slot.roll()
        self.assertIsInstance(slot.obj, Pet)

    def test_shop_slot_food(self):
        slot = ShopSlot("food")
        slot.obj = Food("apple")
        slot.roll()
        self.assertIsInstance(slot.obj, Food)

    def test_shop_level_up(self):
        slot = ShopSlot("levelup")
        tier = slot.obj.tier
        self.assertEqual(tier, 2)

    def test_max_shop(self):
        s = Shop(turn=11)
        s.freeze(0)
        ref_state = s[0].state
        for index in range(10):
            s.roll()
        self.assertEqual(ref_state, s[0].state)

    def test_rabbit_buy_food(self):
        test_player = Player(shop=["honey"], team=["rabbit"])
        start_health = 2
        self.assertEqual(test_player.team[0].pet.health, start_health)

        test_player.buy_food(0, 0)
        expected_end_health = start_health + 1
        self.assertEqual(test_player.team[0].pet.health, expected_end_health)

    def test_empty_shop_from_state(self):
        pet = Pet("fish")
        orig_shop = Shop(slots=[pet])
        orig_shop.buy(pet)
        self.assertEqual(len(orig_shop.slots), 0)

        copy_shop = Shop.from_state(orig_shop.state)
        self.assertEqual(len(copy_shop.slots), 0)

    def test_combine_scorpions(self):
        player = Player(team=["scorpion", "scorpion"])
        player.combine(0, 1)

    def test_combine_coconut_shield(self):
        gorilla = Pet("gorilla")
        gorilla.status = "status-coconut-shield"
        gorilla2 = Pet("gorilla")
        gorilla2.status = "status-melon-armor"
        player = Player(team=[gorilla, gorilla2])
        player.combine(1, 0)
        self.assertEqual(
            gorilla.status, "status-coconut-shield"
        )  # same priority, therefore pet-to-keep keeps its status

    def test_shop_follows_turn_rules(self):
        """Shop slot counts and pet tiers must follow the rules of each turn"""
        player = Player()
        for _ in range(10):
            player.start_turn()
            rules = get_shop_rules(player.shop.turn)
            pslots, fslots, tier_avail = rules[0], rules[1], rules[2]
            self.assertEqual(player.shop.pslots, pslots)
            self.assertEqual(player.shop.fslots, fslots)
            self.assertEqual(len(player.shop.pets), pslots)
            self.assertEqual(len(player.shop.foods), fslots)
            for slot in player.shop:
                self.assertEqual(slot.turn, player.shop.turn)

            ### Roll enough to see every available tier in the shop
            tiers_seen = set()
            for _ in range(100):
                player.shop.roll()
                tiers_seen |= {pet.tier for pet in player.shop.pets}
            self.assertEqual(max(tiers_seen), tier_avail)

    def test_next_turn_updates_shop(self):
        shop = Shop()
        for _ in range(8):
            shop.next_turn()
        rules = get_shop_rules(shop.turn)
        self.assertEqual(len(shop), rules[0] + rules[1])
        self.assertEqual(shop.nslots, rules[0] + rules[1])
        self.assertTrue(all(slot.turn == shop.turn for slot in shop))

    def test_fixed_shop_next_turn(self):
        """Shops built from explicit slots keep their slot layout"""
        player = Player(shop=Shop(["apple", "ant"]))
        player.start_turn()
        self.assertEqual(len(player.shop), 2)
        self.assertEqual(len(player.shop.pets), 1)
        self.assertEqual(len(player.shop.foods), 1)

    def test_squirrel(self):
        player = Player(team=Team([Pet("squirrel")]))
        player.start_turn()
        self.assertEqual(player.shop[3].cost, 2)

        player.roll()
        self.assertEqual(player.shop[3].cost, 3)

    def test_pill_1gold(self):
        player = Player(shop=Shop(["sleeping-pill"]), team=Team(["fish"]))
        player.buy_food(0, 0)
        self.assertEqual(player.gold, 9)

    def test_cupcake(self):
        player = Player(shop=Shop(["cupcake"]), team=Team([Pet("fish")]))

        player.buy_food(0, 0)
        self.assertEqual(player.team[0].attack, 5)  # fish 2/3
        self.assertEqual(player.team[0].health, 6)

        player.end_turn()
        player.start_turn()

        self.assertEqual(player.team[0].attack, 2)
        self.assertEqual(player.team[0].health, 3)

    def test_apple(self):
        player = Player(shop=Shop(["apple"]), team=Team([Pet("beaver")]))

        player.buy_food(0, 0)
        self.assertEqual(player.team[0].attack, 4)
        self.assertEqual(player.team[0].health, 3)

    def test_shop_levelup_from_combine(self):
        player = Player(shop=Shop(["fish", "fish"]), team=Team([Pet("fish")]))
        player.buy_combine(1, 0)
        player.buy_combine(0, 0)
        self.assertEqual(len(player.shop), 1)

    def test_shop_levelup_from_food(self):
        pet = Pet("fish")
        pet.level = 2
        pet.experience = 2
        player = Player(shop=Shop(["ant", "ant", "ant", "chocolate"]), team=[pet])
        player.buy_food(3, 0)
        self.assertEqual(pet.level, 3)
        ### Chocolate was bought and a level-up pet was added to the shop
        self.assertEqual(len(player.shop.filled), 4)
        levelup = [x for x in player.shop if x.slot_type == "levelup"]
        self.assertEqual(len(levelup), 1)
        self.assertEqual(levelup[0].obj.tier, 2)

    def test_buy_multi_target_food(self):
        player = Player(shop=["sushi"], team=["seal", "rabbit", "fish"])
        player.buy_food(0)
        ### Every pet eats sushi. Seal gives its friends +1 attack when it eats
        ###   and rabbit gives each pet that eats +1 health.
        self.assertEqual(player.team[0].attack, 4)  # 3 + sushi
        self.assertEqual(player.team[0].health, 10)  # 8 + sushi + rabbit
        self.assertEqual(player.team[1].attack, 3)  # 1 + sushi + seal
        self.assertEqual(player.team[1].health, 4)  # 2 + sushi + rabbit
        self.assertEqual(player.team[2].attack, 4)  # 2 + sushi + seal
        self.assertEqual(player.team[2].health, 5)  # 3 + sushi + rabbit

    def test_buy_chocolate(self):
        player = Player(shop=["chocolate"], team=["seal", "rabbit", "fish"])
        player.buy_food(0, 0)
        self.assertEqual(player.team[0].pet.experience, 1)
        self.assertEqual(player.team[0].attack, 3)  # 3
        self.assertEqual(player.team[0].health, 9)  # 8 + rabbit
        self.assertEqual(player.team[1].attack, 2)  # 1 + seal
        self.assertEqual(player.team[1].health, 2)  # 2
        self.assertEqual(player.team[2].attack, 3)  # 2 + seal
        self.assertEqual(player.team[2].health, 3)  # 3

    def test_buy_apple(self):
        player = Player(shop=["apple"], team=["seal", "rabbit", "fish"])
        player.buy_food(0, 0)
        self.assertEqual(player.team[0].attack, 4)  # 3 + apple
        self.assertEqual(player.team[0].health, 10)  # 8 + apple + rabbit
        self.assertEqual(player.team[1].attack, 2)  # 1 + seal
        self.assertEqual(player.team[1].health, 2)  # 2
        self.assertEqual(player.team[2].attack, 3)  # 2 + seal
        self.assertEqual(player.team[2].health, 3)  # 3

    def test_canned_food(self):
        state = np.random.RandomState(seed=1).get_state()
        player = Player(
            shop=Shop(["fish", "canned-food"], seed_state=state), team=["fish"]
        )
        player.buy_food(1)

        def base(slot):
            return Pet(slot.obj.name).attack, Pet(slot.obj.name).health

        ### check immediate result: fish 2/3 + canned food 1/1
        self.assertEqual(player.shop[0].obj.attack, 3)
        self.assertEqual(player.shop[0].obj.health, 4)

        ### check result after 1 roll
        player.roll()
        attack, health = base(player.shop[0])
        self.assertEqual(player.shop[0].obj.attack, attack + 1)
        self.assertEqual(player.shop[0].obj.health, health + 1)

        ### check result in a new turn
        player.end_turn()
        player.start_turn()
        for slot in player.shop.pets:
            self.assertEqual(slot.attack, Pet(slot.name).attack + 1)
            self.assertEqual(slot.health, Pet(slot.name).health + 1)


# %%
