/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Planner;

// Import de la superclase
import core.player.AbstractPlayer;

// Import de utiles
import core.game.StateObservation;
import ontology.Types;
import tools.ElapsedCpuTimer;
import ontology.Types;

import java.io.IOException;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

/**
 * Agent class that extends the AbstractPlayer class.
 * This class represents an agent who uses a planner in order to find the
 * correct actions so that it can get to its goal.
 * @author vladislav
 */
public class Agent extends AbstractPlayer {
    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer) {}

    @Override
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        
        // Strings that containt the paths for the planner, the domain file,
        // the problem file and the log file
        String plannerRoute = "../planners/Metric-FF/ff",
               domainFile = "../planners/Metric-FF/Ej7dominio.pddl",
               problemFile = "../planners/Metric-FF/Ej7problema1.pddl",
               logFileRoute = "../log";
        
        // Creater new process which will run the planner
        ProcessBuilder pb = new ProcessBuilder(plannerRoute, "-o", domainFile,
                "-f", problemFile);
        File log = new File(logFileRoute);
        
        // Clear log file
        try {
            PrintWriter writer = new PrintWriter(log);
            writer.print("");
            writer.close();
        } catch (FileNotFoundException ex) {
            System.out.println("Error: archivo no encontrado " + ex);
        }
        
        
        // Redirect error and output streams
        pb.redirectErrorStream(true);
        pb.redirectOutput(ProcessBuilder.Redirect.appendTo(log));
        
        // Run process and wait until it finishes
        try {
            Process process = pb.start();
            try {
                process.waitFor();
            } catch (InterruptedException inte) {
                System.out.println("Se ha interumpido el proceso");
            }            
        } catch (IOException e) {
            System.out.println("Se ha producido una excepcion IOException: " + e);
        }
        
        return Types.ACTIONS.ACTION_NIL;
    }
}
