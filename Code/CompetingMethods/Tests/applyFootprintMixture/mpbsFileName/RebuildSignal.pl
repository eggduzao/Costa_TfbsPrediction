my $K = 6;
my $HK = $K / 2;
my %KMerBias = ();
my @Seqs, @RCSeqs;
my @Bases = ( 'A', 'C', 'G', 'T' );
#my @Kmers = @Bases;
my @Signal;
my @Subset;

open(N, $ARGV[0]) or die "Cannot open bias file";


while ( <N> )
{
	@line = split(/\s+/);
	$KMerBias{$line[0]} = $line[1];
}
close(N);

open(N, $ARGV[1]) or die "Cannot open fasta file";

while ( <N> )
{
	if ( m/^[ACGTacgt]/ )
	{
		chomp;
		push(@Seqs,$_);
		push(@RCSeqs,RevComp($_));
	}

}
close(N);

if ( $ARGV[2] ne "" )
{
	open(N, $ARGV[2]) or die "Cannot open subset file";
	while ( <N> )
	{
		chomp;
		push(@Subset,$_);
	}
	close(N);
}

#print scalar(@Subset);

$SeqC = scalar(@Seqs);
$SeqL = length($Seqs[0]);
#print $SeqL . "\n";

for ( $i = 0; $i < $SeqL; $i++ )
{
	$Signal[$i] = 0;
}


if ( scalar(@Subset) == 0 )
{
	for ( $k = 0; $k < $SeqC; $k++ )
	{
		for ( $i = $HK; $i < ($SeqL-$HK); $i++ )
		{
			$Signal[$i] = $Signal[$i] + $KMerBias{substr($Seqs[$k],$i-$HK,$K)};
			$Signal[$i] = $Signal[$i] + $KMerBias{substr($RCSeqs[$k],($SeqL-1)-($i+$HK),$K)};
		}
	}
}
else
{
	for ( $k = 0; $k < scalar(@Subset); $k++ )
	{
		for ( $i = $HK; $i < ($SeqL-$HK); $i++ )
		{
			$Signal[$i] = $Signal[$i] + $KMerBias{substr($Seqs[$Subset[$k]],$i-$HK,$K)};
			$Signal[$i] = $Signal[$i] + $KMerBias{substr($RCSeqs[$Subset[$k]],($SeqL-1)-($i+$HK),$K)};
		}
	}
}



#print scalar(@Signal) . "\n";

#for ( $i = 0; $i < $SeqL; $i++ )
for ( $i = $HK; $i < ($SeqL-$HK); $i++ )
{
	print "$Signal[$i] ";
}
print "\n";



sub RevComp {
        my $dna = $_[0];
        my $revcomp = reverse($dna);
        $revcomp =~ tr/ACGTacgt/TGCAtgca/;
        $revcomp;
}

