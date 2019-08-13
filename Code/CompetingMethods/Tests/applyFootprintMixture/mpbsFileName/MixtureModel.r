MultMTrain <- function(Cuts)
{
	BPar <- colSums(Cuts) / sum(Cuts);
	UPar <- rep ( 1 / ncol(Cuts), ncol(Cuts) );
	return ( list(BPar=BPar,UPar=UPar) );
}


MultMMixture_Full <- function( TF_Bed=c(), Cuts=c(), peakbed=c(), Pos_Ind=c(), DHS_Ind=c(), Neg_Ind=c(), 
	PadLen=25, Collapse=TRUE, Fixed = FALSE, Iterations = 1, k = 2, ReturnPar = F,
	Background = "Uniform", FastaName = "", Plot = F  )
{
	TF_Len <- TF_Bed[1,3] - TF_Bed[1,2] + 1;
	
	
	if ( is.null(Pos_Ind) )
	{
		Pos_Ind <- unique((IntersectBeds(TF_Bed,peakbed))[,1]);
	}
	if ( is.null(Neg_Ind) )
	{
		Neg_Ind <- setdiff( c(1:nrow(TF_Bed)), Pos_Ind );
	}
	#Initialize

	if ( length(Pos_Ind) < 1 )
	{	print("Not sufficent number of binding sites to train from");	}

	#Initial Parameters
	Init_T <- c();
	#Break positives into samples
	if ( length(Pos_Ind) < (k-1) )
	{	print("Too many mixture parameters, reduce 'k'" );	}

	Sep_I <- sample(c(1:k-1),length(Pos_Ind),replace=T);
	for ( i in 1:(k-1) )
	{
		Init_T <- rbind(Init_T,MultMTrain(Cuts[Pos_Ind[Sep_I==i],])$BPar);
	}
	if ( Background == "Seq" && FastaName != "" )
	{
		Init_T <- rbind(Init_T,as.numeric(BuildSeqBiasBackground(FastaName)) );
	}
	else
	{
		Init_T <- rbind(Init_T,rep(1/ncol(Cuts),ncol(Cuts)) );
	}


	if ( Plot )
	{
		x11(); 
		par(mfrow=c(2,1));
		plot(x=c(1:ncol(Cuts)),y=Init_T[1,],type="l",col=2,ylim=c(min(Init_T),max(Init_T)),
			xlab="",ylab="", lwd = 2 );
		#legend("topleft",c("Footprint","Background"),lwd=rep(2,k),col=c(2:(k+1)));
		abline(v=c((PadLen+1),(ncol(Cuts)-PadLen)));
		for ( i in 2:k )
		{
			lines(x=c(1:ncol(Cuts)),y=Init_T[i,],type="l",col=i+1,lwd=2);
		}	
	}


	if ( Fixed & k == 2)
	{
		M <- MultMixture(Cuts[c(Pos_Ind,Neg_Ind),],t=Init_T,k=k, fixed=c(FALSE,TRUE));
		flatTheta <- 2;
	}
	else
	{
		M <- MultMixtureFast(Cuts[c(Pos_Ind,Neg_Ind),],t=Init_T,k=k);
		TFBSmass <- rowSums( M$theta[, (PadLen+1):(ncol(Cuts)-PadLen)])
		flatTheta <- order(TFBSmass,decreasing=T)[1];
	}
	

	if ( Plot )
	{
		plot(x=c(1:ncol(Cuts)),y=M$theta[1,],type="l",col=2,ylim=c(min(M$theta),max(M$theta)),
			xlab="",ylab="", lwd = 2 );
		legend("topleft",c("Footprint","Background"),lwd=rep(2,k),col=c(2:(k+1)));
		abline(v=c((PadLen+1),(ncol(Cuts)-PadLen)));
		for ( i in 2:k )
		{
			lines(x=c(1:ncol(Cuts)),y=M$theta[i,],type="l",col=i+1,lwd=2);
		}	
	}

	
	z <- c();
	for ( j in 1:k )
	{
		z <- cbind( z, apply(as.matrix(Cuts),1,dmultinom,prob=M$theta[j,]) );
	}
	z <- z / rowSums(z);
		
	fpTheta <- setdiff(c(1:k),flatTheta)

	#Return FP Models / BG Model, likelihood ratio
	if ( k > 2 )
	{	llr <- rowSums(z[,fpTheta]) / z[,flatTheta];	}

	#Return FP Model / BG Model, likelihood ratio
	else
	{	llr <- z[,fpTheta] / z[,flatTheta];	}

	llr[which(llr==Inf)] <- .Machine$double.xmax;
	llr[which(llr==-Inf)] <- .Machine$double.xmin;
	llr <- log(llr);
	
	if ( Plot )
	{
		x11();
		plot(x=c(1:ncol(Cuts)),y=M$theta[1,]
			,type="l",ylim=c(min(M$theta),max(M$theta)),
			xlab="",ylab="", lwd=2, col="blue" );
		abline(v=c((PadLen+1),(ncol(Cuts)-PadLen)));
		for ( i in 2:k )
		{
			lines(x=c(1:ncol(Cuts)),y=M$theta[i,],type="l",col=i+1,lwd=2);
		}	
		lines(x=c(1:ncol(Cuts)),y=M$theta[flatTheta,],type="l",col="red",lwd=2);
	}


	pars <- list(theta=M$theta,UI=flatTheta,loglik=M$loglik);
	if ( ReturnPar )
	{	return(list(par=pars,llr=llr,z=z));	}
	else
	{	return(llr);	}

}


MultMixtureFast <- function(Cuts, k = 2, a = c(), t = c() )
{
	Zsize <- nrow(Cuts);
	Xsum <- rowSums(Cuts);
	X <- Cuts;
	Mix <- multmixEM(as.matrix(Cuts),k=k,theta=t,lambda=a);
	BIC <- -2 * Mix$loglik + ( ( length(Mix$theta) - k ) + ( k - 1 ) ) * log(nrow(Cuts));
	return(list(theta=Mix$theta,loglik=Mix$loglik,lambda=Mix$lambda));

}


MultMixture <- function(Cuts, k = 2, a = c(), t = c(), fixed = c() )
{
	Zsize <- nrow(Cuts);
	Xsum <- rowSums(Cuts);
	X <- Cuts;
	Data <- list();

	if ( is.null(fixed) )
	{
		fixed = rep(FALSE,k);
	}
	if ( is.null(a) )
	{
		a <- rdirichlet(1,rep(1,k));
	}
	
	if ( is.null(t) )
	{	
		z <- sample(c(1:k),Zsize,replace=T);
		for ( i in 1:k )
		{
			t <- rbind(t, MultMTrain( X[which(z==i),] )$BPar );
		}
	}

	#Find samples that have large spikes 
	z <- c();
	for ( j in 1:k )
	{
		z <- cbind(z,apply(X,1,dmultinom,prob=t[j,]));
	}
	Zero_I <- which(rowSums(z)==0);
	#Set those samples to 0
	if ( length(Zero_I) > 0 )
	{
		print(paste("Removed" , length(Zero_I), "irregular samples"));
		X[Zero_I,] <- rep(0,ncol(Cuts));
	}
	z <- z / rowSums(z);
	

	Loglik <- c();		
	for ( j in 1:k )
	{
		Loglik <- cbind(Loglik, a[j] * apply(X,1,dmultinom,prob=t[j,]) );
	}
	Loglik <- sum(log(rowSums(Loglik)));




	flag <- TRUE;
	I <- 1;
	epsilon <- 1e-08;
	while ( flag )
	{
		z <- c();
		#print("Calculate Z");
		
		for ( j in 1:k )
		{
			z <- cbind(z,apply(X,1,dmultinom,prob=t[j,]));
		}
		#Zero_I <- which(rowSums(z)==0);
		z <- z / rowSums(z);
		
	
		#z[Zero_I,] <- rep(1/k,k);
		
		#print("Reestimate theta");
		for ( i in 1:k )
		{
			if ( ! fixed[i] )
			{
				for ( j in 1:ncol(X) )
				{
					t[i,j] = sum( X[,j] * z[,i] ) / sum( Xsum * z[,i] );
				}
			}
		}	
				
		#print("Reestimate mixture weight");
		for ( j in 1:k )
		{
			a[j] <- sum(z[,j]) / sum(z);
		}
		
		
		NewLoglik <- c();		
		for ( j in 1:k )
		{
			NewLoglik <- cbind(NewLoglik, a[j] * apply(X,1,dmultinom,prob=t[j,]) );
		}
		NewLoglik <- sum(log(rowSums(NewLoglik)));

		#print(c(Loglik,NewLoglik));
		#print(NewLoglik - Loglik);

		
		#if ( NewLoglik > Loglik )
		if ( NewLoglik - Loglik > epsilon )
		{
			Loglik <- NewLoglik;
			Data[[I]] <- list(z=z,t=t,c=Cuts,a=a,loglik=Loglik);
			I <- I + 1;
		}
		else
		{
			flag <- FALSE;
		}
	}

	return(list(theta=Data[[length(Data)]]$t,loglik=Data[[length(Data)]]$loglik,lambda=a));
}

BuildSeqBiasBackground <- function(FastaName)
{
	BiasFile <- "SeqBias.txt";
	inFA <- FastaName;
	outSignal <- "signal.txt";
	system( paste("perl RebuildSignal.pl ",
		BiasFile, " ", inFA, " > ", outSignal, sep="" ) );
	signal <- read.table(outSignal);
	M <- signal / sum(signal);
	return(M);
}

IntersectBeds <- function( Bed1, Bed2 )
{
	gr1 <- GRanges( Rle(Bed1[,1]), IRanges(start=Bed1[,2],end=Bed1[,3] ) );
	gr2 <- GRanges( Rle(Bed2[,1]), IRanges(start=Bed2[,2],end=Bed2[,3] ) );
	M <- findOverlaps(gr1,gr2);
	M <- as.matrix(M);
	return(M);
}

