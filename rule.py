# coding: utf-8

import sys
import math


#######################################
##          Consts
#######################################

# Cells
BLUEBERRIES_CRATE = "B"
STRAWBERRIES_CRATE = "S"
ICE_CREAM_CRATE = "I"
DOUGH_CRATE = "H"

WINDOW = "W"
EMPTY_TABLE = "#"
DISHWASHER = "D"
FLOOR_CELL = "."
CHOPPING_BOARD = "C"
OVEN = "O"

# Items
NONE = "NONE"
DISH = "DISH"
ICE_CREAM = "ICE_CREAM"
BLUEBERRIES = "BLUEBERRIES"
STRAWBERRIES = "STRAWBERRIES"
CHOPPED_STRAWBERRIES = "CHOPPED_STRAWBERRIES"
DOUGH = "DOUGH"
CROISSANT = "CROISSANT"
RAW_TART = "RAW_TART"
TART = "TART"

# Supplies
SUPPLIES = {
  DISHWASHER: DISH,
  BLUEBERRIES_CRATE: BLUEBERRIES,
  ICE_CREAM_CRATE: ICE_CREAM,
  STRAWBERRIES_CRATE: STRAWBERRIES,
  DOUGH_CRATE: DOUGH,
}


#######################################
##          Utils 
#######################################
def log(*x):
    print(x, file=sys.stderr)

def argmax(_list):
  if not _list:
    return None
  _list = [(i, x) for i, x in enumerate(_list)]
  return max(_list, key=lambda x: x[1])[0]

def argmin(_list):
  if not _list:
    return None
  _list = [(i, x) for i, x in enumerate(_list)]
  return min(_list, key=lambda x: x[1])[0]

def flatten(_list):
  new_list = []
  for l in _list:
    new_list += l
  #map(lambda l: new_list += l, _list)
  return new_list

class UnitBase(object):
  @property
  def items(self):
    return [x for x in self.item.split('-') if x]

  @property
  def hasItem(self):
    return False if self.item == NONE else True

class Player(UnitBase):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.item = ""

class Tile(UnitBase):
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name
        self.item = SUPPLIES[name] if name in SUPPLIES else ""

    def parse_name(self):
        return self.name.split("-")

    def __repr__(self):
        return "(%d, %d, %s)" % (self.x, self.y, str(self.items))

class Order(UnitBase):
  def __init__(self, item, award):
    self.item = item
    self.award = award

  def __repr__(self):
    return "%s" % (str(self.items))

class Game:
  def __init__(self):
    self.player = Player()
    self.partner = Player()
    self.tiles = []
    self.orders = []
    self.oven_timer = None

  def addTile(self, x, y, tileChar):
    if tileChar != '.':
      self.tiles.append(Tile(x, y, tileChar))

  def getTileByName(self, name):
    for t in self.tiles:
      if t.name == name:
        return t
    log('Invalid Tile Name !!!')

  def getTilesByItem(self, item):
    return [t for t in self.tiles if item in t.items]

  def getTileByCoords(self, x, y):
    for t in self.tiles:
      if t.x == x and t.y == y:
        return t

  def updatePlayer(self, x, y, item):
    self.player.x = x
    self.player.y = y
    self.player.item = item 

  def updatePartner(self, x, y, item):
    self.partner.x = x
    self.partner.y = y
    self.partner.item = item

  def updateOrders(self, orders):
    self.orders = orders

  def use(self, tile):
    return "USE %d %d" % (tile.x, tile.y) + "; Python Starter AI"

  def move(self, tile):
    return "MOVE %d %d" % (tile.x, tile.y)

  def wait(self):
    return "WAIT"


class LogicBase:
  pass

class RuleBasedLogic(LogicBase):
  def getAction(self, game):
    log('orders', game.orders)
    log('holding', game.player.items)
    log('oven', game.getTileByName(OVEN).item)
    log('dishes', game.getTilesByItem(DISH))
    log('croissant', game.getTilesByItem(CROISSANT))
    log('chopped_straw', game.getTilesByItem(CHOPPED_STRAWBERRIES))

    order = self.decide_order(game.orders)
    if order:
      unprepared_items = [item for item in order.items if item not in game.player.items]
      if unprepared_items:
        if CROISSANT in unprepared_items:
          log('Creating a croissant')
          croissant_tiles = game.getTilesByItem(CROISSANT)
          if croissant_tiles:
            log('taking a croissant at %s' % str(croissant_tiles[0]))
            return game.use(croissant_tiles[0])
          elif game.getTileByName(OVEN).item == DOUGH:
            log('waiting to finish burning')
            return game.wait()
          elif not game.player.hasItem:
            log('Taking a dough')
            return game.use(game.getTileByName(DOUGH_CRATE))
          elif DOUGH in game.player.items:
            log('delivering a dough to oven')
            return game.use(game.getTileByName(OVEN))

        elif CHOPPED_STRAWBERRIES in unprepared_items:
          log('Creating chopped strawberries')
          if CHOPPED_STRAWBERRIES in unprepared_items:
            chopped_straw_tiles = game.getTilesByItem(CHOPPED_STRAWBERRIES)
            if chopped_straw_tiles:
              log('taking a chopped_straw at %s' % str(chopped_straw_tiles[0]))
              return game.use(chopped_straw_tiles[0])
            elif not game.player.hasItem:
              log('Taking raw strawberries')
              return game.use(game.getTileByName(STRAWBERRIES_CRATE))
            elif STRAWBERRIES in game.player.items:
              log('Going to chopping board with strawberries')
              return game.use(game.getTileByName(CHOPPING_BOARD))
          if DISH not in game.player.items:
            return self.search_dish(game, order.items)
        else:
          log('Preparing others in order than chopped_strawberries and croisssant')
          tiles = flatten([game.getTilesByItem(item) for item in unprepared_items])
          idx = argmin([self.calc_distance(game.player, tile) for tile in tiles])
          if idx is not None: 
            tile = tiles[idx]
            return game.use(tile)
      else:
        log('Delivering the dish')
        return game.use(game.getTileByName(WINDOW))

    else: # Prepare croissant and chopped_strawberries for partner.
      required_items = []
      for order in game.orders:
        required_items += order.items
      required_items = set(required_items)

      n_croissant = game.getTilesByItem(CROISSANT)
      n_chopped_strawberries = game.getTilesByItem(CHOPPED_STRAWBERRIES)

      if CROISSANT in game.player.items or CHOPPED_STRAWBERRIES in game.player.items:
        log('putting created croissant or chopped strawberries to a table nearby')
        empty_table = self.search_empty_table_around(game)
        if empty_table:
          return game.use(empty_table)
        else:
          return game.move(game.getTileByName(WINDOW))

      if n_croissant < n_chopped_strawberries:
        log('Creating a croissant for partner')
        if game.getTileByName(OVEN).item == DOUGH:
          log('waiting to finish burning')
          return game.wait()
        elif game.getTileByName(OVEN).item == CROISSANT:
          log('taking croissant from oven')
        elif not game.player.hasItem:
          log('Taking a dough')
          return game.use(game.getTileByName(DOUGH_CRATE))
        elif DOUGH in game.player.items:
          log('delivering a dough to oven')
          return game.use(game.getTileByName(OVEN))

      else:
        if not game.player.hasItem:
          log('Taking raw strawberries')
          return game.use(game.getTileByName(STRAWBERRIES_CRATE))
        elif STRAWBERRIES in game.player.items:
          log('Going to chopping board with strawberries')
          return game.use(game.getTileByName(CHOPPING_BOARD))

      return game.wait()

  def decide_order(self, orders):
    for order in orders:
      if CHOPPED_STRAWBERRIES not in order.items and CROISSANT not in order.items:
        return order
      elif CHOPPED_STRAWBERRIES not in order.items or CROISSANT not in order.items:
        return order

  def calc_distance(self, player, tile):
    return 0

  def search_dish(self, game, items_in_order):
    dish_tiles = set(game.getTilesByItem(DISH))
    if len(dish_tiles) > 3:
      dish_tiles -= game.getTilesByName(DISHWASHER)
    items_in_order = set(items_in_order)

    dish_tiles = sorted([(tile, len(set(tile.items).intersection(items_in_order))) for tile in dish_tiles], key=lambda x:-x[1])
    dish_tiles = [tile for tile, _ in dish_tiles]
    log('dish priorities', dish_tiles)
    for tile in dish_tiles:
      if not set(tile.items) - items_in_order:
        return game.use(tile)

  def search_empty_table_around(self, game):
    x = game.player.x
    y = game.player.y
    for tile in reversed(game.tiles):
      if tile.name != EMPTY_TABLE or tile.item:
        continue
      if tile.x < x - 1 or tile.x > x + 1:
        continue
      if tile.y < y - 1 or tile.y > y + 1:
        continue
      return tile
    
game = Game()
logic = RuleBasedLogic()

# ALL CUSTOMERS INPUT: to ignore until bronze
num_all_customers = int(input())
for i in range(num_all_customers):
  # customer_item: the food the customer is waiting for
  # customer_award: the number of points awarded for delivering the food
  customer_item, customer_award = input().split()
  customer_award = int(customer_award)

# KITCHEN INPUT
for y in range(7):
  kitchen_line = input()
  for x, tileChar in enumerate(kitchen_line):
    game.addTile(x, y, tileChar)


# game loop
while True:
  turns_remaining = int(input())

  # PLAYERS INPUT
  #Gather and update player information
  player_x, player_y, player_item = input().split()
  player_x = int(player_x)
  player_y = int(player_y)
  game.updatePlayer(player_x, player_y, player_item)

  #Gather and update partner information
  partner_x, partner_y, partner_item = input().split()
  partner_x = int(partner_x)
  partner_y = int(partner_y)
  game.updatePartner(partner_x, partner_y, partner_item)

  # #Gather and update table information

  num_tables_with_items = int(input())  # the number of tables in the kitchen that currently hold an item

  for t in game.tiles:
    if t.name == EMPTY_TABLE:
      t.item = NONE

  for i in range(num_tables_with_items):
    table_x, table_y, item = input().split()
    table_x = int(table_x)
    table_y = int(table_y)
    game.getTileByCoords(table_x, table_y).item = item

  # oven_contents: ignore until bronze league
  oven_contents, oven_timer = input().split()
  oven_timer = int(oven_timer)
  game.getTileByName(OVEN).item = oven_contents
  game.oven_timer = oven_timer

  num_customers = int(input())  # the number of customers currently waiting for food
  orders = []
  for i in range(num_customers):
    customer_item, customer_award = input().split()
    customer_award = int(customer_award)
    orders.append(Order(customer_item, customer_award))

  game.updateOrders(orders)
  # GAME LOGIC
  #Gather plate & Icecream
  action = logic.getAction(game)
  if not action:
    action = "WAIT"
  print(action)
  log('action', action)

