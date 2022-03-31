#!usr/bin/perl -w
$mag = "2"; 
$num = 0;

$dir = "../REAL_result_30day_sel";

$together = "phase_allday.txt";
#Actually, it is the hypoDD input format	
open(OT,">$together");
	my @files = <$dir/phase_sel_*.txt>;
    foreach my $file (@files) {
        open(JK,"<$file");  
        @par = <JK>;
        close(JK);

        print $file . "\n";
    	foreach $file(@par){
            ($test,$jk) = split(' ',$file);
            if($test =~ /^\d+$/){
                ($jk,$year1,$mon1,$dd1,$time,$ot,$std,$lat,$lon,$dep) = split(' ',,$file);
                ($hour,$min,$sec) = split('\:',$time);
                $num++;
                print OT "# $year1  $mon1  $dd1   $hour    $min    $sec    $lat    $lon    $dep     $mag     0.0     0.0    0.0   $num\n";
            }else{
                ($net,$station,$phase,$traveltime,$pick,$amplitude) = split(' ',$file);
                print OT "$station $pick 1 $phase\n";
            }
        }
    
    }
	
close(OT);





