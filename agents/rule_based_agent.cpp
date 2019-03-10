#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <map>

using namespace std;

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
    if (!item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}


int findString( vector<string> m_array,  string value )
{
    auto iter = std::find( m_array.begin(), m_array.end(), value);
    size_t index = std::distance( m_array.begin(), iter );
    if(index == m_array.size())
    {
        return -1;
    }
    return index;
}

const string DISH = "DISH";
const string BLUEBERRIES = "BLUEBERRIES";
const string ICE_CREAM = "ICE_CREAM";
const string STRAWBERRIES = "STRAWBERRIES";
const string CHOPPED_STRAWBERRIES = "CHOPPED_STRAWBERRIES";
const string CHOPPED_DOUGH = "CHOPPED_DOUGH";
const string CROISSANT = "CROISSANT";
const string DOUGH = "DOUGH";
const string WINDOW = "WINDOW";
const string TART = "TART";
const string RAW_TART = "RAW_TART";

const vector<string> SUPPLY_ITEMS = {DISH, BLUEBERRIES, ICE_CREAM, STRAWBERRIES, WINDOW};


class Player{
private:
    int x;
    int y;
    string item;
    vector<string> itemVector;
public:
    Player(int x, int y, string item){
        this->x = x;
        this->y = y;
        this->item = item;
        if(item == "NONE"){

        }
        else{
            this->itemVector = split(item, '-');
        }
    }

    vector<string> getItemVector(){
        return itemVector;
    }
    string getItem(){
        return item;
    }
    int getX(){
        return x;
    }
    int getY(){
        return y;
    }
};

class TableItem{
private:
    int x;
    int y;
    string item;
    vector<string> itemVector;
public:
    TableItem(int x, int y, string item){
        this->x = x;
        this->y = y;
        this->item = item;
        this->itemVector = split(item, '-');
    }

    string getItem(){
        return item;
    }
    vector<string> getItemVector(){
        return itemVector;
    }
    int getX(){
        return x;
    }
    int getY(){
        return y;
    }
};

class Customer{
private:
    string item;
    int award;
    vector<string> itemVector;
public:
    Customer(string item, int award){
        this->item = item;
        this->award = award;
        this->itemVector = split(item, '-');
    }
    vector<string> getItemVector(){
        return itemVector;
    }
     string getItem(){
        return item;
    }
};

map<string, char> FOOD_NAME_TO_CHAR;
vector<Customer*> allCustomers;
vector<vector<char> > kitchenMap;

vector<TableItem*> searchTableItem(vector<TableItem*> tableItemVector, string item){
    vector<TableItem*> result;
    for(TableItem* table : tableItemVector){
        if(table->getItem() == item){
            result.push_back(table);
        }
    }
    return result;
}

int* searchKitchen(vector<vector<char> > kitchenMap, char item){
    static int result[2];
    //cerr << "searchKitchenfor:" << item << endl;
    for(int y = 0; y < kitchenMap.size(); y ++){
        for(int x = 0; x < kitchenMap[y].size(); x ++){
            if(kitchenMap[y][x] == item){
                result[0] = x; result[1] = y;
                //cerr << "found" << x << "/" << y << endl;
                return result;
            }
        }
    }
    cerr << "not found!" << endl;
    return result;
}

vector<int*> searchKitchenByStringName(vector<vector<char> > kitchenMap, vector<TableItem*> tableItems, string item){
    int dish_count = 0;
    for(TableItem* tableItem : tableItems){
        if(findString(tableItem->getItemVector(), "DISH") != -1){
            dish_count ++;
        }
    }

    vector<int*> result;
    //cerr << findString(SUPPLY_ITEMS, item) << endl;
    if(findString(SUPPLY_ITEMS, item) != -1){
        if(dish_count == 3 && item == "DISH"){

        }
        else{
            int* resultPos = searchKitchen(kitchenMap, FOOD_NAME_TO_CHAR[item]);
            result.push_back(resultPos);
        }
    }
    vector<TableItem*> resultTableItems = searchTableItem(tableItems, item);
    for(TableItem* tableItem : resultTableItems){
        static int pos[2];
        pos[0] = tableItem->getX(); pos[1] = tableItem->getY();
        result.push_back(pos);
    }
    return result;
}

vector<int*> searchAdjacentEmptyTable(vector<vector<char> > kitchenMap, vector<TableItem*> tableItemVector, int playerX, int playerY){
    vector<int*> result;
    for(int y = 0; y < kitchenMap.size(); y ++){
        for(int x = 0; x < kitchenMap[y].size(); x ++){
            if(kitchenMap[y][x] == '#' && abs(x - playerX) <= 1 && abs(y - playerY) <= 1){
                bool item_found = false;
                for(TableItem* table : tableItemVector){
                    if(table->getX() == x && table->getY() == y){
                        item_found = true;
                        break;
                    }
                }
                if(!item_found){
                    static int pos[2];
                    pos[0] = x; pos[1] = y;
                    result.push_back(pos);
                }
            }
        }
    }
    return result;
}

int* searchNearestEmptyTable(vector<vector<char> > kitchenMap, vector<TableItem*> tableItemVector, int playerX, int playerY){
    static int result[2];
    int min_dist = 999;
    for(int y = 0; y < kitchenMap.size(); y ++){
        for(int x = 0; x < kitchenMap[y].size(); x ++){
            if(kitchenMap[y][x] == '#'){
                bool item_found = false;
                for(TableItem* table : tableItemVector){
                    if(table->getX() == x && table->getY() == y){
                        item_found = true;
                        break;
                    }
                }
                if(!item_found){
                    if( abs(x - playerX) +  abs(y - playerY) < min_dist){
                        min_dist = abs(x - playerX) +  abs(y - playerY);
                        result[0] = x; result[1] = y;
                    }
                }
            }
        }
    }
    return result;
}


void initialize(){
    FOOD_NAME_TO_CHAR[DISH] = 'D';
    FOOD_NAME_TO_CHAR[BLUEBERRIES] = 'B';
    FOOD_NAME_TO_CHAR[ICE_CREAM] = 'I';
    FOOD_NAME_TO_CHAR[STRAWBERRIES] = 'S';
    FOOD_NAME_TO_CHAR[DOUGH] = 'H';
    FOOD_NAME_TO_CHAR[WINDOW] = 'W';

    
    int numAllCustomers;
    cin >> numAllCustomers; cin.ignore();
    for (int i = 0; i < numAllCustomers; i++) {
        string customerItem; // the food the customer is waiting for
        int customerAward; // the number of points awarded for delivering the food
        cin >> customerItem >> customerAward; cin.ignore();
        allCustomers.push_back(new Customer(customerItem, customerAward));
    }

    for (int i = 0; i < 7; i++) {
        string kitchenLine;
        getline(cin, kitchenLine);
        vector<char> kithenLineVec;
        for(int n = 0; n < (int)kitchenLine.size(); ++n){
            char ch = kitchenLine[n];
            if(ch == '0' || ch == '1'){
                kithenLineVec.push_back('.');
            }
            else{
                kithenLineVec.push_back(ch);
            }
            
        }
        kitchenMap.push_back(kithenLineVec);
    }
}

Player* player;
Player* partner;
vector<TableItem*> tableItems;
vector<Customer*> currentCustomers;

string ovenContents; // ignore until wood 1 league
int ovenTimer;

void parseInput(){
    tableItems.clear();
    currentCustomers.clear();

    int turnsRemaining;
    cin >> turnsRemaining; cin.ignore();
    int playerX;
    int playerY;
    string playerItem;
    cin >> playerX >> playerY >> playerItem; cin.ignore();
    player = new Player(playerX, playerY, playerItem);
    int partnerX;
    int partnerY;
    string partnerItem;
    cin >> partnerX >> partnerY >> partnerItem; cin.ignore();
    partner = new Player(partnerX, partnerY, partnerItem);
    
    int numTablesWithItems; // the number of tables in the kitchen that currently hold an item
    cin >> numTablesWithItems; cin.ignore();
    for (int i = 0; i < numTablesWithItems; i++) {
        int tableX;
        int tableY;
        string item;
        cin >> tableX >> tableY >> item; cin.ignore();
        tableItems.push_back(new TableItem(tableX, tableY, item));
    }
   
    cin >> ovenContents >> ovenTimer; cin.ignore();
    
    int numCustomers; // the number of customers currently waiting for food
    cin >> numCustomers; cin.ignore();
    for (int i = 0; i < numCustomers; i++) {
        string customerItem;
        int customerAward;
        cin >> customerItem >> customerAward; cin.ignore();
        currentCustomers.push_back(new Customer(customerItem, customerAward));
    }
}

// turncount, nextx, nexty
int* getRequiredItemPickupTurn(int currentX, int currentY, string item){
    vector<int*> positions = searchKitchenByStringName(kitchenMap, tableItems, item);
    static int* result = new int[3];
    result[0] = 9999;
    for(int* pos : positions){
        int kitchen[kitchenMap.size()][kitchenMap[0].size()];
        for(int y = 0; y < kitchenMap.size(); y ++){
            for(int x = 0; x < kitchenMap[y].size(); x ++){
                 kitchen[y][x] = 999;
            }
        }

        int target = 0;
        kitchen[currentY][currentX] = 0;
        int min_x;
        int min_y;
        while(true){
            bool found_flag = false;
            for(int target_y = max(pos[1]-1, 0); target_y < min(pos[1]+2, 7); target_y ++){
                for(int target_x = max(pos[0]-1, 0); target_x < min(pos[0]+2,11); target_x ++){
                    if(kitchen[target_y][target_x] != 999){
                        min_x = target_x;
                        min_y = target_y;
                        found_flag = true;  
                    }
                }
            }

            /*for(int y = 0; y < kitchenMap.size(); y ++){
                for(int x = 0; x < kitchenMap[y].size(); x ++){
                    if(kitchen[y][x] == 999){
                        cerr << "*";
                    }
                    else{
                        cerr << kitchen[y][x];
                    }
                }
                cerr << endl;
            }*/

            if(found_flag){
                break;
            }

            for(int y = 0; y < kitchenMap.size(); y ++){
                for(int x = 0; x < kitchenMap[y].size(); x ++){
                    if(kitchen[y][x] == target){
                        for(int target_y = y-1; target_y < y+2; target_y ++){
                            for(int target_x = x-1; target_x < x+2; target_x ++){
                                if(abs(target_y - y) != 0 && abs(target_x - x) != 0){
                                    continue;
                                }
                                if(kitchenMap[target_y][target_x] == '.' && kitchen[target_y][target_x] == 999){
                                    kitchen[target_y][target_x] = target + 1;
                                }
                            }
                        }
                    }
                }
            }
            target ++;
        }

        cerr << "player:" << currentX << ":" << currentY << endl;
        cerr << "pos:" << pos[0] << ":" << pos[1] << endl;
        cerr << "min:" << min_x << ":" << min_y << endl;
        cerr << "turn:" << target << endl;

        if(result[0] > (target + 3) / 4 + 1){
            result[0] = (target + 3) / 4 + 1;
            result[1] = min_x;
            result[2] = min_y;
        }
    }
    if(result[0] > 9000){
        cerr << "result error!" << endl;
        cerr << item << endl;
    }
    return result;
}


// TODO use table combined items
int calculateRequiredTurnRec(vector<string> required_items, vector<bool> item_flags, int currentX, int currentY, int currentTurn){
    bool item_left = false;
    int min_turn = 99999;
    for(int i = 0; i < required_items.size(); i++){
        if(!item_flags[i]){
            item_left = true;
            item_flags[i] = true;
            cerr << required_items.size() << endl;
            cerr << required_items[i] << endl;
            int* next_info = getRequiredItemPickupTurn(currentX, currentY, required_items[i]);
            int required_turn = calculateRequiredTurnRec(required_items, item_flags, next_info[1], next_info[2], currentTurn + next_info[0]);
            min_turn = min(required_turn, min_turn);
            item_flags[i] = false;
        }
    }
    if(!item_left){
        currentTurn += getRequiredItemPickupTurn(currentX, currentY, WINDOW)[0];
        return currentTurn;
    }
    return min_turn;
}

int calculateRequiredTurn(Customer* customer, Player* player){
    int menu_start_index = 0;
    vector<string> required_items;
    vector<bool> item_flags;
    for(string item : customer->getItemVector()){
        if(findString(player->getItemVector(), item) == -1){
            vector<int*> positions = searchKitchenByStringName(kitchenMap, tableItems, item);
            if(positions.size() == 0){
                return -1;
            }
            required_items.push_back(item);
            item_flags.push_back(false);
        }
    }
    for(string item : player->getItemVector()){
        if(findString(customer->getItemVector(), item) == -1){
            // requires trashing for making target food
            return -1;
        }
    }

    // TODO not considering making parts
    // TODO assume that dish is first pick
    if(required_items.size() == 0){
        return getRequiredItemPickupTurn(player->getX(), player->getY(), WINDOW)[0];
    }
    else if(required_items[0] == "DISH"){
        item_flags[0] = true;
        int* next_info = getRequiredItemPickupTurn(player->getX(), player->getY(), required_items[0]);
        return calculateRequiredTurnRec(required_items, item_flags, next_info[1], next_info[2], next_info[0]);
    }
    else{
        return calculateRequiredTurnRec(required_items, item_flags, player->getX(), player->getY(), 0);
    }
}

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
int main()
{
    initialize();
    // game loop
    while (1) {
       
        parseInput();

        int require_chopped_strawberries= 0;
        int require_croissant = 0;
        int require_tart = 0;

        //debug output
        cerr << "player_item:" <<  player->getItem() << endl;
        cerr << "oven_contents:" << ovenContents << endl;
        int min_turn = 9999;
        Customer* target_customer = nullptr;
        for(Customer* customer : currentCustomers){
            for(string item : customer->getItemVector()){
                if(item == CHOPPED_STRAWBERRIES){
                    require_chopped_strawberries ++;
                }
                if(item == CROISSANT){
                    require_croissant ++;
                }
                if(item == TART){
                    require_tart ++;
                }
            }

            int turn = calculateRequiredTurn(customer, player);
            cerr << "order:" << customer->getItem() << endl;
            cerr << "turn:" << turn << endl;
            if(turn != -1 && min_turn > turn){
                min_turn = turn;
                target_customer = customer;
            }
        }

        for(int y = 0; y < kitchenMap.size(); y ++){
            for(int x = 0; x < kitchenMap[y].size(); x ++){
                cerr << kitchenMap[y][x];
            }
            cerr << endl;
        }

        require_chopped_strawberries = min(require_chopped_strawberries, 2);
        require_croissant = min(require_croissant, 1);
        require_tart = min(require_tart, 1);

        // Write an action using cout. DON'T FORGET THE "<< endl"
        // To debug: cerr << "Debug messages..." << endl;

        // need choping strawberry?

        bool customerNeedStrawberry = findString(currentCustomers[0]->getItemVector(), CHOPPED_STRAWBERRIES) != -1;
        bool hasRawStrawberry = findString( player->getItemVector(), STRAWBERRIES) != -1;
        bool hasChoppedStrawberry = findString( player->getItemVector(), CHOPPED_STRAWBERRIES) != -1;
        bool hasDish = findString( player->getItemVector(), DISH) != -1;
        bool hasCroissant = findString( player->getItemVector(), CROISSANT) != -1;
        bool hasTart = findString( player->getItemVector(), TART) != -1;
        bool hasRawTart = findString( player->getItemVector(), RAW_TART) != -1;
        bool hasDough = findString( player->getItemVector(), DOUGH) != -1;
        bool hasChoppedDough = findString( player->getItemVector(), CHOPPED_DOUGH) != -1;

        // make two choppped strawberry in global
        // TODO make stte machene?

        // calculate required turn for each order


        // chop strawberry
        if(hasRawStrawberry){
            int* pos = searchKitchen(kitchenMap, 'C');
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        }
        else if(hasChoppedDough){
            int* pos = searchKitchen(kitchenMap, FOOD_NAME_TO_CHAR[BLUEBERRIES]);
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        }
        // chop dough
        else if(hasDough && searchTableItem(tableItems, TART).size() < require_tart){
            int* pos = searchKitchen(kitchenMap, 'C');
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        }
        // put chipped strawberry
        else if(hasChoppedStrawberry && !hasDish){
            int* emptyTables = searchNearestEmptyTable(kitchenMap, tableItems, player->getX(), player->getY());
            cout << "USE " << emptyTables[0] << " " << emptyTables[1] << endl;
        }
        // pick strawberry to chop
        else if(player->getItemVector().size() == 0 && searchTableItem(tableItems, CHOPPED_STRAWBERRIES).size() < require_chopped_strawberries){
            int* pos = searchKitchen(kitchenMap, FOOD_NAME_TO_CHAR[STRAWBERRIES]);
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        }
        // put croissant
        else if(hasCroissant && !hasDish){
            int* emptyTables = searchNearestEmptyTable(kitchenMap, tableItems, player->getX(), player->getY());
            cout << "USE " << emptyTables[0] << " " << emptyTables[1] << endl;
        } 
        // put tart
        else if(hasTart && !hasDish){
            int* emptyTables = searchNearestEmptyTable(kitchenMap, tableItems, player->getX(), player->getY());
            cout << "USE " << emptyTables[0] << " " << emptyTables[1] << endl;
        } 
        // bake dough
        else if(hasDough){
            int* pos = searchKitchen(kitchenMap, 'O');
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        } 
        // bake raw tart
        else if(hasRawTart){
            int* pos = searchKitchen(kitchenMap, 'O');
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        } 
        // wait oven
        else if(ovenContents != "NONE" && hasDough){
            cout << "WAIT" << endl;
        } 
        else if(ovenContents == "NONE" && player->getItemVector().size() == 0 && searchTableItem(tableItems, CROISSANT).size() < require_croissant){
            int* pos = searchKitchen(kitchenMap, FOOD_NAME_TO_CHAR[DOUGH]);
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        } 
        else if(ovenContents == "NONE" && player->getItemVector().size() == 0 && searchTableItem(tableItems, TART).size() < require_tart){
            int* pos = searchKitchen(kitchenMap, FOOD_NAME_TO_CHAR[DOUGH]);
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        } 
        // pick croissant hurry!
        else if(ovenContents == "TART" && player->getItemVector().size() == 0){
            int* pos = searchKitchen(kitchenMap, 'O');
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        } 
        // pick croissant hurry!
        else if(ovenContents == "CROISSANT" && player->getItemVector().size() == 0){
            int* pos = searchKitchen(kitchenMap, 'O');
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        } 
        // wait to bake
        else if(ovenContents == "DOUGH" && player->getItemVector().size() == 0){
             int* pos = searchKitchen(kitchenMap, 'O');
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        } 
        else if(ovenContents == "RAW_TART" && player->getItemVector().size() == 0){
             int* pos = searchKitchen(kitchenMap, 'O');
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        } 
        // get next target item
        else if(target_customer != nullptr){
            // TODO use nearest logic
            vector<string> required_items;
            for(string item : target_customer->getItemVector()){
                if(findString(player->getItemVector(), item) == -1){
                    required_items.push_back(item);
                }
            }
            bool dispose = false;
            for(string item : player->getItemVector()){
                if(findString(target_customer->getItemVector(), item) == -1){
                    int* emptyTables = searchNearestEmptyTable(kitchenMap, tableItems, player->getX(), player->getY());
                    cout << "USE " << emptyTables[0] << " " << emptyTables[1] << endl;
                    dispose = true;
                    break;
                }
            }
            if(!dispose){
                if(required_items.size() == 0){
                    int* pos = searchKitchen(kitchenMap, 'W');
                    cout << "USE " << pos[0] << " " << pos[1] << endl;
                }
                else{
                    vector<int*> pos = searchKitchenByStringName(kitchenMap, tableItems, required_items[0]);
                    cout << "USE " << pos[0][0] << " " << pos[0][1] << endl;
                }
            }
        }
        else {
            cout << "WAIT" << endl;
        }
        /*else if(currentCustomers[0]->getItemVector().size() > player->getItemVector().size()){
            string targetItem = currentCustomers[0]->getItemVector()[player->getItemVector().size()];
            cerr << "targetItem:" <<  targetItem << endl;
            int* pos = searchKitchenByStringName(kitchenMap, tableItems, targetItem)[0];
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        }
        // order is completed
        else{
            int* pos = searchKitchen(kitchenMap, 'W');
            cout << "USE " << pos[0] << " " << pos[1] << endl;
        }*/
    }
}