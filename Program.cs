using System;
using DDR;
using System.Collections.Generic;

//This is demo for selective prediction of soccer match outcome for British Premier League.
//It uses Divisive Data Resorting explained in https://arxiv.org/abs/2104.01714
//The deterministic component in ensemble explained in https://www.youtube.com/watch?v=eS_k6L638k0
//and also in two articles https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//and https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742

//Authors of the concept Andrew Polar and Mike Poluektov
//this code is written by Andrew Polar.

namespace PremierLeague7
{
    class Program
    {
        static double Bet2Money(int bet)
        {
            if (bet < 0)
            {
                return 100.0 * 100.0 / Math.Abs((double)bet);
            }
            else
            {
                return bet;
            }
        }

        static void MakePrediction(Ensemble ensemble, double[,] initial_matrix, Dictionary<string, int> standings, string fileName)
        {
            DataReader dr = new DataReader();
            List<Game> currentlist = dr.ProcessFile(fileName);

            double[,] matrix = new double[17, 17];
            for (int i = 0; i < 17; ++i)
            {
                for (int j = 0; j < 17; ++j)
                {
                    matrix[i, j] = initial_matrix[i, j];
                }
            }

            int Total = 0;
            int Right = 0;
            double Amount = 0.0;
            Console.WriteLine("Prediction for British Premier League");
            foreach (Game current in currentlist)
            {
                int indexHome = -1;
                int indexAway = -1;

                if (standings.ContainsKey(current.teamHome) && standings.ContainsKey(current.teamAway))
                {
                    indexHome = standings[current.teamHome];
                    indexAway = standings[current.teamAway];
                }

                if (indexHome < 0 || indexAway < 0)
                {
                    continue;
                }

                string modeldata = Static.GetInputOutputVector(matrix, indexHome, indexAway);
                string[] textinputsandoutput = modeldata.Split(",");
                if (7 != textinputsandoutput.Length)
                {
                    Console.WriteLine("wrong string length");
                    Environment.Exit(0);
                }
                double d = 0;
                List<double> inputsonly = new List<double>();
                for (int i = 0; i < 6; ++i)
                {
                    bool res = Double.TryParse(textinputsandoutput[i], out d);
                    if (false == res)
                    {
                        Console.WriteLine("TryParse failed");
                        Environment.Exit(0);
                    }
                    inputsonly.Add(d);
                }

                double[] sample = ensemble.GetOutput(inputsonly.ToArray());

                int NH = 0;
                int ND = 0;
                int NA = 0;
                for (int i = 0; i < sample.Length; ++i)  
                {
                    if (0 == (int)(Math.Round(sample[i]))) ++ND;
                    else if (Math.Round(sample[i]) < 0.0) ++NA;
                    else if (Math.Round(sample[i]) > 0.0) ++NH;
                }

                int diff = current.goalsHome - current.goalsAway;

                double PH = (double)NH / (double)(sample.Length);
                double PD = (double)ND / (double)(sample.Length);
                double PA = (double)NA / (double)(sample.Length);

                double MH = PH * Bet2Money(current.Bet1) - (1.0 - PH) * 100.0;
                double MD = PD * Bet2Money(current.BetX) - (1.0 - PD) * 100.0;
                double MA = PA * Bet2Money(current.Bet2) - (1.0 - PA) * 100.0;

                string predicted_result = "";
                if (MH > MA && MH > MD)
                {
                    if (diff > 0)
                    {
                        Amount += Bet2Money(current.Bet1);
                        ++Right;
                        predicted_result = "right prediction";
                    }
                    else
                    {
                        Amount -= 100.0;
                        predicted_result = "wrong prediction";
                    }
                    ++Total;
                }
                if (MA > MH && MA > MD)
                {
                    if (diff < 0)
                    {
                        Amount += Bet2Money(current.Bet2);
                        ++Right;
                        predicted_result = "right prediction";
                    }
                    else
                    {
                        Amount -= 100.0;
                        predicted_result = "wrong prediction";
                    }
                    ++Total;
                }
                if (MD > MA && MD > MH)
                {
                    //Prediction of draws is risky, it gives positive balance, but reduces winning amount
                    //if (0 == diff)
                    //{
                    //    Amount += Bet2Money(current.BetX);
                    //    ++Right;
                    //    predicted_result = "right prediction";
                    //}
                    //else
                    //{
                    //    Amount -= 100.0;
                    //    predicted_result = "wrong prediction";
                    //}
                    //++Total;

                    //betting of draw is replaced by either home or away
                    if (MH > MA)
                    {
                        if (diff > 0)
                        {
                            Amount += Bet2Money(current.Bet1);
                            ++Right;
                            predicted_result = "right prediction";
                        }
                        else
                        {
                            Amount -= 100.0;
                            predicted_result = "wrong prediction";
                        }
                        ++Total;
                    }
                    if (MA > MH)
                    {
                        if (diff < 0)
                        {
                            Amount += Bet2Money(current.Bet2);
                            ++Right;
                            predicted_result = "right prediction";
                        }
                        else
                        {
                            Amount -= 100.0;
                            predicted_result = "wrong prediction";
                        }
                        ++Total;
                    }
                }

                Console.WriteLine("{0}. {1}, {2}", Total, current.GetGameInfo(), predicted_result);

                matrix[indexHome, indexAway] = diff;
            }

            Console.WriteLine("\nTotal bets {0}", Total);
            Console.WriteLine("Right predictions {0}", Right);
            Console.WriteLine("Total betting amount ${0}, $100 per game", Total * 100);
            Console.WriteLine("The ending balance, the amount obtained or lost (lost when negative) ${0:0.00}", Amount);
            Console.WriteLine("Winning to betting amount ratio {0:0.00}", Amount / (double)(Total) / 100.0);
        }

        static void Main(string[] args)
        {
            Static.MakeTrainingData();

            //Resort data
            Resorter resorter = new Resorter();
            resorter.ReadData(@"..\..\..\data\Big-Training-Data.csv");
            Console.WriteLine("Data is read");

            //in order to compare DDR ensemble to simple bagging, comment out the block below down to LABEL
            resorter.Resort(1);
            resorter.Resort(2);
            resorter.Resort(3);
            resorter.Resort(5);
            resorter.Resort(7);
            resorter.Resort(11);
            resorter.Resort(13);
            Console.WriteLine("Resorting is finished");
            //LABEL

            Ensemble ensemble = new Ensemble(Resorter._inputs, Resorter._target);
            ensemble.BuildModels(23);

            MakePrediction(ensemble, Static._matrix_2019_2020, Static._standings_2019_2020, "../../../DATA/2020-2021.txt");
        }
    }
}

