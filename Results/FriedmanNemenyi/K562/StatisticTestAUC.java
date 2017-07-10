import java.io.*;
import java.util.*;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;


public class StatisticTestAUC {

	public static void main(String []args){


		String []arquivos = {
		    "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/FriedmanNemenyi/K562/Input/auc.txt"
		};


		for(String arq : arquivos ){
			try {
				double [][]data;
				BufferedReader reader = new BufferedReader(new FileReader(new File(arq)));
				String line = reader.readLine();
				int lin = 0, col = 0;
				//System.out.println(line);
				StringTokenizer st = new StringTokenizer(line,"\t");
				data = new double[Integer.parseInt(st.nextToken())][Integer.parseInt(st.nextToken())];
				boolean asc = st.nextToken().equals("asc");
				while((line = reader.readLine())!= null){
				//System.out.println(line);
				   st = new StringTokenizer(line,"\t");
				   while(st.hasMoreElements()){
							 String token = st.nextToken();
							 //System.out.println(token);
							 data[lin][col] = Float.parseFloat(token);
							 col++;
				   }
				   lin++;
				   col = 0;
				}
				System.out.println( "Estatisticas -- "+arq+"("+asc+")\n" + StatisticTestAUC.estatisticas(data, asc) );

				System.out.println( "Win/Tie/Loss -- "+arq+"\n" + StatisticTestAUC.tabelaDesempenho(data, asc) );
			}catch(Exception e){e.printStackTrace(); System.out.println("ARQUIVO: "+arq); return;}
		}

	}
	
	/**
	 * Make the Friedman and Nemenyi tests for a results instances set
	 * @param data
	 * @param asc
	 * @return
	 */
	public static String estatisticas(Instances dataResults, boolean asc){
		StringBuffer statsResults = new StringBuffer();
		double F_Friedman = Double.NaN;
		int nl =0 , nc =0;
		try{
			//remove std deviations and dataset names from data
			Instances data = new Instances(dataResults);
			
			data.setClassIndex(-1);
			for(int i=data.numAttributes()-1;i>=0;i-=2){
				data.deleteAttributeAt(i);
			}		
					
			nl = data.numInstances();
			nc = data.numAttributes();
			double [][]ranks = new double[nl][nc];  
			for(int i=0;i<nl;++i){
				double []line = data.instance(i).toDoubleArray();
				int []sorts = Utils.sort(line);			
				for(int j=0;j<nc;++j){
					if(!asc){
						ranks[i][sorts[j]] = j+1;
					}else{
						ranks[i][sorts[j]] = nc-j;
					}				
				}
				BitSet modificados = new BitSet(nc);
				modificados.clear();
				for(int j=0;j<nc;++j){
					if(modificados.get(j)) continue;
					double val = data.instance(i).value(j);
					int iguais =1;
					double soma= ranks[i][j];
					
					for(int jj=j+1;jj<nc;++jj){
						
						if(Utils.eq(data.instance(i).value(jj), val)){
							++iguais;
							soma += ranks[i][jj];
						}
					}				
					if(iguais > 1)
					for(int jj=j;jj<nc;++jj){
						if(Utils.eq(data.instance(i).value(jj), val)){						
							modificados.set(jj);
							ranks[i][jj] = soma / ((double)iguais);
						}
					}
					
				}
			}
			
			
			double []medRank = new double[nc];
			for(int j=0;j<nc;++j){	
				medRank[j] = 0;
				for(int i=0;i<nl;++i){
					medRank[j] += ranks[i][j];
				}
				medRank[j] /= nl;			
			}
			
			double chiFriedman =  (12*nl)/(nc*(nc+1));
			double tmp = 0;
			for(int j=0;j<nc;++j){			
				
				tmp += medRank[j] * medRank[j];
			}
			chiFriedman *= (tmp - (nc*((nc+1)*(nc+1))/4));
			F_Friedman = ((nl-1)*chiFriedman)/((nl*(nc-1))-chiFriedman);
			double pValue_Friedman = Double.NaN;
			
		
			pValue_Friedman = weka.core.Statistics.FProbability(F_Friedman, nc-1, (nc-1)*(nl-1));
		
			double []qAlpha5pct = {1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164};
			double []qAlpha10pct = {1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920};
			double critDiff = Math.sqrt((nc*(nc+1.0))/(6.0*nl));
			
			 
			statsResults.append("\n\n\n\n\\subsubsection{Testes sobre "+data.relationName()+"}:\n\n");
			statsResults.append("\n\n\\textbf{Friedman teste pvalue} (FF="+F_Friedman+"): "+pValue_Friedman+".\n\n");
			statsResults.append("\n\n\\textbf{Nemenyi Test}:\n\n");
			int []sorts = Utils.sort(medRank);
			statsResults.append("CritDiff(.05) "+(qAlpha5pct[nc-2]*critDiff)+".\n\n\n");
			statsResults.append("CritDiff(.10) "+(qAlpha10pct[nc-2]*critDiff)+".\n\n\n");
			for(int j=0;j<nc;++j){						
				for(int jj=j+1;jj<nc;++jj){
					//statsResults.append("C"+sorts[nc-j-1]+"/C"+sorts[nc-jj-1]+" "+ (medRank[sorts[nc-j-1]]-medRank[sorts[nc-jj-1]]));				
					statsResults.append("\\textbf{"+data.attribute(sorts[nc-j-1]).name()+"/"+
							data.attribute(sorts[nc-jj-1]).name()+"} "+ (medRank[sorts[nc-j-1]]-medRank[sorts[nc-jj-1]]+".\n\n"));
				}
			}
			statsResults.append("\n\n\n\n");
		}catch(Exception e){
			System.out.println("erro calculando pvalue Friedman: " +
					"nc="+nc+"--nl="+nl+"--FF="+F_Friedman
					+"\n"+e.getMessage());
			e.printStackTrace();
		}
		return statsResults.toString();
	}
	
	
	
	public static String estatisticas(double [][]matriz, boolean asc){
        StringBuffer statsResults = new StringBuffer(); 
		int nl = matriz.length;
		int nc = matriz[0].length;
		double [][]ranks = new double[nl][nc];  
		for(int i=0;i<nl;++i){
			int []sorts = Utils.sort(matriz[i]);			
			for(int j=0;j<nc;++j){
				if(!asc){
					ranks[i][sorts[j]] = j+1;
				}else{
					ranks[i][sorts[j]] = nc - j;
				}				
			}
			BitSet modificados = new BitSet(nc);
			modificados.clear();
			for(int j=0;j<nc;++j){
				if(modificados.get(j)) continue;
				double val = matriz[i][j];
				int iguais =1;
				double soma= ranks[i][j];
				
				for(int jj=j+1;jj<nc;++jj){
					if(matriz[i][jj] == val){
						++iguais;
						soma += ranks[i][jj];
					}
				}				
				if(iguais > 1)
				for(int jj=j;jj<nc;++jj){
					if(matriz[i][jj] == val){						
						modificados.set(jj);
						ranks[i][jj] = soma / ((double)iguais);
					}
				}
				
			}
		}
		
		double []medRank = new double[nc];
		for(int j=0;j<nc;++j){	
			medRank[j] = 0;
			for(int i=0;i<nl;++i){
				medRank[j] += ranks[i][j];
			}
			medRank[j] /= nl;			
		}
		int []orderedRank = Utils.sort(medRank);
		statsResults.append("Ranking:\n");
		for(int j=0;j<nc;++j){
			statsResults.append(orderedRank[j]+"\t"+medRank[ orderedRank[j] ]+"\n");
		}

		double chiFriedman =  (12.0*nl)/(nc*(nc+1));
		double tmp = 0;
		for(int j=0;j<nc;++j){
			
			tmp += medRank[j] * medRank[j];
		}
		chiFriedman *= (tmp - (nc*(((nc+1)*(nc+1))/4)));
		double F_Friedman = ((nl-1)*chiFriedman)/((nl*(nc-1))-chiFriedman);
		System.out.println("ChiFriedman: "+ chiFriedman
				+" F_Friedman: "+F_Friedman+" df1: "+(nc-1)+" df2:"+ (nc-1)*(nl-1));
		double pValue_Friedman = weka.core.Statistics.FProbability(F_Friedman, nc-1, (nc-1)*(nl-1));
		//double []qAlpha5pct = {1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164};
		//double []qAlpha10pct = {1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920};
        double []qAlpha5pct = {1.960, 2.344, 2.569, 2.728, 2.850, 2.948, 3.031, 3.102, 3.164, 3.219, 3.268, 3.313, 3.354, 3.391,
        3.426, 3.458, 3.489, 3.517, 3.544, 3.569, 3.593, 3.616, 3.637, 3.658, 3.678, 3.696, 3.714, 3.732};
        double []qAlpha10pct = {1.645, 2.052, 2.291, 2.460, 2.589, 2.693, 2.780, 2.855, 2.920, 2.978, 3.030, 3.077, 3.120, 3.159,
        3.196, 3.230, 3.261, 3.291, 3.319, 3.346, 3.371, 3.394, 3.417, 3.439, 3.459, 3.479, 3.498, 3.516};
		double critDiff = Math.sqrt((nc*(nc+1.0))/(6.0*nl));
		
		
		statsResults.append("Friedman teste pvalue: "+pValue_Friedman+"\n\n");
		statsResults.append("Nemenyi Test:");
		int []sorts = Utils.sort(medRank);
		statsResults.append("CritDiff(.05) "+(qAlpha5pct[nc-2]*critDiff)+"\n");
		statsResults.append("CritDiff(.10) "+(qAlpha10pct[nc-2]*critDiff)+"\n");
		for(int j=0;j<nc;++j){						
			for(int jj=j+1;jj<nc;++jj){
                //statsResults.append("C"+sorts[nc-j-1]+"/C"+sorts[nc-jj-1]+" "+ (medRank[sorts[nc-j-1]]-medRank[sorts[nc-jj-1]])+"\n");
				double diff = medRank[sorts[nc-j-1]]-medRank[sorts[nc-jj-1]];
				if ( diff >= (qAlpha10pct[nc-2]*critDiff)){
					statsResults.append(sorts[nc-j-1]+" "+sorts[nc-jj-1]+" "+ (medRank[sorts[nc-j-1]]-medRank[sorts[nc-jj-1]])+"\n");
				}
			}
		}


		//monta "tabela" de comparacao par a par
		statsResults.append("\n\nTabela de Comparacao (+ significa diferenca estatistica 90% e * 95%)\n,");

		for(int j=0;j<nc;++j){
			statsResults.append(j+",");
		}
		/**
		 * algoritmo da coluna jj melhor que da linha j com x% de confianca
		 */
		for(int j=0;j<nc;++j){
			statsResults.append("\n"+j+",");
			for(int jj=0;jj<nc;++jj){
				double diff = medRank[j]-medRank[jj];
				if ( diff >= (qAlpha10pct[nc-2]*critDiff) && diff < (qAlpha5pct[nc-2]*critDiff)){
					statsResults.append("+,");
				}else if(diff >= (qAlpha5pct[nc-2]*critDiff)){
					statsResults.append("*,");
				}else{
					statsResults.append(",");
				}
			}
		}




		return statsResults.toString();
	}

	/**
	 * Monta uma tabela de Vitorias/Empates/Derrotas
	 * @param data dados, algoritmos em colunas, bases nas linhas
	 * @param asc ascendente ou descendente
	 * @return
	 */
	public static String tabelaDesempenho(double[][] data, boolean asc) {
		StringBuffer output = new StringBuffer();
		int numAlgoritmos = data[0].length;
		int numBases = data.length;
		int [][]vitorias = new int[ numAlgoritmos ][ numAlgoritmos ];
		int [][]empates  = new int[ numAlgoritmos ][ numAlgoritmos ];
		int [][]derrotas = new int[ numAlgoritmos ][ numAlgoritmos ];

		for(int i=0; i< numAlgoritmos; ++i){

			for(int j=0;j< numAlgoritmos; ++j){
				if( i == j ){
					continue;
				}

				for(int b=0; b< numBases; ++b){
					if(data[b][i] > data[b][j]){
						if(asc){
							++vitorias[i][j];
						}else{
							++derrotas[i][j];
						}
					}else if(data[b][i] < data[b][j]){
						if(asc){
							++derrotas[i][j];
						}else{
							++vitorias[i][j];
						}
					}else{
						++empates[i][j];
					}
				}
			}
		}

		for(int i=0; i< numAlgoritmos; ++i){
			output.append("\n");
			for(int j=0;j< numAlgoritmos; ++j){
				if(i==j){
					output.append("---,");
				}else{
					output.append(vitorias[i][j]+"/"+empates[i][j]+"/"+derrotas[i][j]+",");
				}
			}
		}
		return output.toString();
	}
	
}
