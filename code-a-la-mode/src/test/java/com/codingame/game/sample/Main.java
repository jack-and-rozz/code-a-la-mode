package com.codingame.game.sample;

import com.codingame.gameengine.runner.MultiplayerGameRunner;
import com.codingame.gameengine.runner.dto.GameResult;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Main {
    // args0... ai1 dir
    // args1... ai1 dir
    // args2... ai1 dir
    public static void main(String[] args) {
        if(args.length != 5){
            System.out.println("usage: ai1 ai2 ai3 resultfile jsonfile");
        }

        MultiplayerGameRunner gameRunner = new MultiplayerGameRunner();

        for(int i = 0; i < 3; i ++){
            if(args[i].equals("naive")){
                gameRunner.addAgent(NaiveAllItemsPlayer.class);
            }
            else if(args[i].equals("insta")){
                gameRunner.addAgent(InstaFoodPlayer.class);
            }
            else{
                gameRunner.addAgent(args[i]);
            }
        }

        gameRunner.setLeagueLevel(4);

        GameResult result = gameRunner.getResult();
        try{
            File file = new File(args[3]);
            FileWriter filewriter = new FileWriter(file);

            filewriter.write(result.scores.get(0) + "\n");
            filewriter.write(result.scores.get(1) + "\n");
            filewriter.write(result.scores.get(2) + "\n");

            filewriter.close();
        }catch(IOException e){
            System.out.println(e);
        }

        String resultJson = gameRunner.getJSONResult();

        try{
            File file = new File(args[4]);
            FileWriter filewriter = new FileWriter(file);

            filewriter.write(resultJson);

            filewriter.close();
        }catch(IOException e){
            System.out.println(e);
        }
        //gameRunner.start();
    }
}
