/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */ 
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/* developed at:                                               */
/*                                                             */
/*      Speech Vision and Robotics group                       */
/*      Cambridge University Engineering Department            */
/*      http://svr-www.eng.cam.ac.uk/                          */
/*                                                             */
/*      Entropic Cambridge Research Laboratory                 */
/*      (now part of Microsoft)                                */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright: Microsoft Corporation                    */
/*          1995-2000 Redmond, Washington USA                  */
/*                    http://www.microsoft.com                 */
/*                                                             */
/*              2002  Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*         File: HLlike.c: HMM initialisation program           */
/* ----------------------------------------------------------- */

/* HLlike : is added to HTK  to calculate the log-likelihood of data segments */
/* given a file and its corresponding segmentations. */
/* Amir Harati July 2014  */
/* There is a memory problem with this program. The memory usage growth as we 
   process more files.  The usage should be careful to not go beyond system memory */


char *hrest_version = "!HVER!HLlike:   3.4.1 [CUED 12/03/09]";
char *hrest_vc_id = "$Id: HLlike_space.c,v 1.1 2015/06/18 19:50:18 picone Exp $";


/* Trace Flags */
#define T_TOP    0001    /* Top level tracing */
#define T_LD0    0002    /* File Loading */
#define T_LD1    0004    /* + segments within each file */
#define T_OTP    0010    /* Observation Probabilities */
#define T_ALF    0020    /* Alpha matrices */
#define T_BET    0040    /* Beta matrices */
#define T_OCC    0100    /* Occupation Counters */
#define T_TAC    0200    /* Transition Counters */
#define T_MAC    0400    /* Mean Counters */
#define T_VAC   01000    /* Variance Counters */
#define T_WAC   02000    /* MixWeight Counters */
#define T_TRE   04000    /* Reestimated transition matrix */
#define T_WRE  010000    /* Reestimated mixture weights */
#define T_MRE  020000    /* Reestimated means */
#define T_VRE  040000    /* Reestimated variances */
#define T_LGP 0100000    /* Compare LogP via alpha and beta */


#include "HShell.h"     /* HMM ToolKit Modules */
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HAudio.h"
#include "HWave.h"
#include "HVQ.h"
#include "HParm.h"
#include "HLabel.h"
#include "HModel.h"
#include "HTrain.h"
#include "HUtil.h"


/* Global Settings */
static char * segLab = NULL;     /* segment label if any */
static LabId  segId  = NULL;     /* and its index */
static char * labDir = NULL;     /* label file directory */
static char * labExt = "lab";    /* label file extension */
static char * outDir = NULL;     /* output macro file directory, if any */

static Boolean firstTime = TRUE; /* Flag used to enable InitSegStore */

static FileFormat dff=UNDEFF;    /* data file format */
static FileFormat lff=UNDEFF;    /* label file format */
static float minVar  = 0.0;      /* minimum variance */
static float mixWeightFloor=0.0; /* Floor for mixture weights */
static float tMPruneThresh = 10.0;    /* tied mix prune threshold */
static char *hmmfn;              /* HMM definition file name */
static char *outfn=NULL;         /* output definition file name */
static UPDSet uFlags = (UPDSet) (UPMEANS|UPVARS|UPTRANS|UPMIXES);     /* update flags */
static int  trace    = 0;        /* Trace level */
static ConfParam *cParm[MAXGLOBS];   /* configuration parameters */
static int nParm = 0;               /* total num params */
static Boolean segReject = TRUE; /* Enable short train segment rejection */


/* Global Data Structures */
static HMMSet hset;        /* The current unitary hmm set */
static HLink hmm;          /* link to the hmm itself */
static int nStates;        /* numStates of hmm */
static int nStreams;       /* numStreams of hmm */
static HSetKind hsKind;          /* kind of the HMM system */
static int maxMixes;       /* max num mixtures across all streams */
static int maxMixInS[SMAX];/* array[1..swidth[0]] of max mixes */
static int nSeg;           /* num training segments */
static int nTokUsed;       /* actual number of tokens used */
static int maxT,minT,T;    /* max,min and current segment lengths */
static DMatrix alpha;      /* array[1..nStates][1..maxT] of forward prob */
static DMatrix beta;       /* array[1..nStates][1..maxT] of backward prob */
static Matrix outprob;     /* array[2..nStates-1][1..maxT] of output prob */
static Vector **stroutp;   /* array[1..maxT][2..nStates-1][1..nStreams] ...*/
                           /* ... of streamprob */
static Matrix **mixoutp;   /* array[2..nStates-1][1..maxT][1..nStreams]
                              [1..maxMixes] of mixprob */
static Vector occr;        /* array[1..nStates-1] of occ count for cur time */
static Vector zot;         /* temp storage for zero mean obs vector */
static Vector vFloor[SMAX];      /* variance floor - default is all zero */
static float vDefunct=0.0;       /* variance below which mixture defunct */

static SegStore segStore;        /* Storage for data segments */
static MemHeap segmentStack;     /* Used by segStore */
static MemHeap alphaBetaStack;   /* For storage of alpha and beta probs */
static MemHeap accsStack;        /* For storage of accumulators */
static MemHeap transStack;       /* For storage of transcription */
static MemHeap bufferStack;      /* For storage of buffer */
static ParmBuf pbuf;             /* Currently input parm buffer */

   
/* ------------------ Process Command Line ------------------------- */

/* SetConfParms: set conf parms relevant to HLlike  */
void SetConfParms(void)
{
   int i;
   double d;
   Boolean b;

   nParm = GetConfig("HLlike", TRUE, cParm, MAXGLOBS);
   if (nParm>0) {
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
      /*if (GetConfBool(cParm,nParm,"SAVEBINARY",&b)) saveBinary = b;*/
      if (GetConfFlt(cParm,nParm,"VDEFUNCT",&d)) vDefunct = d;
   }
}

void ReportUsage(void)
{
   printf("\nUSAGE: HLlike [options] hmmFile trainFiles...\n\n");
   printf(" Option                                       Default\n\n");
   printf(" -e f    Set convergence factor epsilon       1.0E-4\n");
   printf(" -i N    Set max iterations to N              20\n");
   printf(" -l s    Set segment label to s               none\n");
   printf(" -m N    Set min segments needed              3\n");
   printf(" -t      Disable short segment rejection      on\n");
   printf(" -u tmvw Update t)rans m)eans v)ars w)ghts    tmvw\n");
   printf(" -v f    Set minimum variance to f            0.0\n");
   printf(" -c f    Tied Mixture pruning threshold       10.0\n");
   printf(" -w f    Set mix wt floor to f x MINMIX       0.0\n");
   PrintStdOpts("BFGHILMSTX");
   printf("\n\n");
}

void SetuFlags(void)
{
   char *s;
   
   s=GetStrArg();
   uFlags=(UPDSet) 0;        
   while (*s != '\0')
      switch (*s++) {
      case 't': uFlags = (UPDSet) (uFlags+UPTRANS); break;
      case 'm': uFlags = (UPDSet) (uFlags+UPMEANS); break;
      case 'v': uFlags = (UPDSet) (uFlags+UPVARS); break;
      case 'w': uFlags = (UPDSet) (uFlags+UPMIXES); break;
      default: HError(2220,"SetuFlags: Unknown update flag %c",*s);
         break;
      }
}

int main(int argc, char *argv[])
{
  int single_file = 0;
   char *datafn, *s;
   void Initialise1(void);
   void Initialise2(void);
   char *replace_string(char *str, char *orig, char *rep);
   void LoadFile(char *fn);
   char *ch;
   char *str1;
   char *ch2;
   char *str2;
   char *labname;
   char *tmp2;
   char *tmp1;
   char *inpLab;

   

   FILE *fp;

   ch=(char*)malloc(1000);
   str1=(char*)malloc(1000);
   ch2=(char*)malloc(1000);
   str2=(char*)malloc(1000);
   labname=(char*)malloc(1000);
   tmp2=(char*)malloc(1000);
   tmp1=(char*)malloc(1000);;
   inpLab=(char*)malloc(1000);
   
   if(InitShell(argc,argv,hrest_version,hrest_vc_id)<SUCCESS)
      HError(2200,"HLlike: InitShell failed");

   InitMem();   InitLabel();
   InitMath();  InitSigP();
   InitWave();  InitAudio();
   InitVQ();    InitModel();
   if(InitParm()<SUCCESS)  
      HError(2200,"HLlike: InitParm failed");

   InitTrain(); InitUtil();

   if (!InfoPrinted() && NumArgs() == 0)
      ReportUsage();
   if (NumArgs() == 0) Exit(0);
   SetConfParms();
   CreateHMMSet(&hset,&gstack,FALSE);
   while (NextArg() == SWITCHARG) {
      s = GetSwtArg();
      if (strlen(s)!=1) 
         HError(2219,"HLlike: Bad switch %s; must be single letter",s);
      switch(s[0]){      
      case 'l':
         if (NextArg() != STRINGARG)
            HError(2219,"HLlike: Segment label expected");
         segLab = GetStrArg();
         break;
      case 'u':
         SetuFlags(); break;
      case 'v':
         minVar = GetChkedFlt(0.0,100.0,s); break;
      case 'c':
         tMPruneThresh = GetChkedFlt(0.0,1000.0,s); break;
      case 'w':
         mixWeightFloor = MINMIX * GetChkedFlt(0.0,10000.0,s); break;
      case 'F':
         if (NextArg() != STRINGARG)
            HError(2219,"HLlike: Data File format expected");
         if((dff = Str2Format(GetStrArg())) == ALIEN)
            HError(-2289,"HLlike: Warning ALIEN Data file format set");
         break;
      case 'G':
         if (NextArg() != STRINGARG)
            HError(2219,"HLlike: Label File format expected");
         if((lff = Str2Format(GetStrArg())) == ALIEN)
            HError(-2289,"HLlike: Warning ALIEN Label file format set");
         break;
      case 'H':
         if (NextArg() != STRINGARG)
            HError(2219,"HLlike: HMM macro file name expected");
         AddMMF(&hset,GetStrArg());
         break;
	 case 'I':
         
	   /* if (NextArg() != STRINGARG)
	     HError(2219,"HRest: MLF file name expected");
	   LoadMasterFile(GetStrArg());
	   from_Labs=0;
	   break;*/
	   HError(2219,"MLF is not supported. Use Lab directory instead");
	    
         break;
      case 'L':
         if (NextArg()!=STRINGARG)
            HError(2219,"HLlike: Label file directory expected");
         labDir = GetStrArg(); break;
	 
      case 'M':
         if (NextArg()!=STRINGARG)
            HError(2219,"HRest: Output macro file directory expected");
         outDir = GetStrArg(); break;  

      case 'T':
         trace = GetChkedInt(0,0100000,s); break;
      case 'X':
         if (NextArg()!=STRINGARG)
            HError(2219,"HLlike: Label file extension expected");
         labExt = GetStrArg(); break;
	 
      case 'i':
	single_file = 1;
	datafn = GetStrArg();
	break;
      default:
         HError(2219,"HLlike: Unknown switch %s",s);
      }
   }
   if (NextArg()!=STRINGARG)
      HError(2219,"HLlike: source HMM file name expected");
   hmmfn = GetStrArg();
   if (outfn==NULL) outfn = hmmfn;
   Initialise1();
   
   do {
     if (NextArg()!=STRINGARG && single_file == 0){
         HError(2219,"HLlike: training data file name expected");
     }
     if (single_file == 0){
       	 datafn = GetStrArg();
     }
      /* load the file */
      LoadFile(datafn);
      
      nSeg = NumSegs(segStore);
      printf("**%ld",nSeg);
      
      ch = strtok(datafn, "/");
      while (ch != NULL) {
	str1 = ch;
	ch = strtok(NULL, "/");
      }
     
      ch2 = strtok(str1,".");
      str2=ch2;
      while (ch2 != NULL) {
	str2=ch2;
	ch2 = strtok(NULL, ".");
      }
      
      strcpy(tmp2,outDir);
      strcat(tmp2,"/");
      
      tmp1 =strcat(str1,".lab");
      strcpy(inpLab,labDir);
      strcat(inpLab,"/");
      strcat(inpLab,tmp1);
      
      labname = strcat(tmp2,tmp1);
     
      fp = fopen(labname,"w");
      
      Initialise2();
      
      compute_LL(fp,inpLab);
      fclose(fp);
      
      /*free(ch);
      free(ch2);
      free(str1);
      free(str2);
      free(labname);
      free(tmp2);
      free(tmp1);
      free(inpLab);
      */
      firstTime=TRUE;
      
   } while (NumArgs()>0);
   
   Exit(0);
   return (0);          /* never reached -- make compiler happy */
}


char *replace_string(char *str, char *orig, char *rep)
{
  static char buffer[4096];
  char *p;

  if(!(p = strstr(str, orig))) 
    return str;

  strncpy(buffer, str, p-str); 
  buffer[p-str] = '\0';

  sprintf(buffer+(p-str), "%s%s", rep, p+strlen(orig));

  return buffer;
}

/* ------------------------ Initialisation ----------------------- */

/* PrintInitialInfo: print a header of program settings */
void PrintInitialInfo(void)
{   

}
   
/* Initialise1: 1st phase of init prior to loading dbase */
void Initialise1(void)
{
   MLink link;
   LabId  hmmId;
   char base[MAXSTRLEN];
   char path[MAXSTRLEN];
   char ext[MAXSTRLEN];
   int s;

   /* Load HMM def */
   if(MakeOneHMM( &hset,BaseOf(hmmfn,base))<SUCCESS)
      HError(2128,"Initialise1: MakeOneHMM failed");
   if(LoadHMMSet( &hset,PathOf(hmmfn,path),ExtnOf(hmmfn,ext))<SUCCESS)
      HError(2128,"Initialise1: LoadHMMSet failed");
   SetParmHMMSet(&hset);
   if (hset.hsKind!=PLAINHS)
      uFlags = (UPDSet) (uFlags & (~(UPMEANS|UPVARS)));

   /* Get a pointer to the physical HMM and set related globals */
   hmmId = GetLabId(base,FALSE);
   link = FindMacroName(&hset,'h',hmmId);
   hmm = (HLink)link->structure;  
   nStates = hmm->numStates;
   nStreams = hset.swidth[0];
   hsKind = hset.hsKind;
   
   /* Stacks for global structures requiring memory allocation */
   CreateHeap(&segmentStack,"SegStore", MSTAK, 1, 0.0, 1000000, LONG_MAX);
   CreateHeap(&alphaBetaStack,"AlphaBetaStore", MSTAK, 1, 0.0, 1000, 1000);
   CreateHeap(&accsStack,"AccsStore", MSTAK, 1, 0.0, 1000, 1000);
   CreateHeap(&transStack,"TransStore", MSTAK, 1, 0.0, 1000, 1000);
   CreateHeap(&bufferStack,"BufferStore", MSTAK, 1, 0.0, 1000, 1000);
   AttachAccs(&hset, &accsStack, uFlags);

   SetVFloor( &hset, vFloor, minVar);

   if(segLab != NULL)
      segId = GetLabId(segLab,TRUE);
   if(trace&T_TOP)
      PrintInitialInfo();

   maxMixes = MaxMixtures(hmm);
   for(s=1; s<=nStreams; s++)
      maxMixInS[s] = MaxMixInS(hmm, s);
   T = maxT = 0; minT = 100000;
}

/* Initialise2: 2nd phase of init after loading dbase */
void Initialise2(void)
{
   int t,j,m,s;

   alpha = CreateDMatrix(&alphaBetaStack,nStates,maxT);
   beta = CreateDMatrix(&alphaBetaStack,nStates,maxT);
   outprob = CreateMatrix(&alphaBetaStack,nStates-1,maxT); /* row 1 not used */
   ZeroMatrix(outprob);
   if (maxMixes>1){
      mixoutp = (Matrix**)New(&alphaBetaStack, (nStates-2)*sizeof(Matrix*));
      mixoutp -= 2;
      for (j=2;j<nStates;j++){
         mixoutp[j] = (Matrix*)New(&alphaBetaStack, maxT*sizeof(Matrix));
         --mixoutp[j];
         for (t=1;t<=maxT;t++){
            mixoutp[j][t] = CreateMatrix(&alphaBetaStack,nStreams,maxMixes);
            for (s=1;s<=nStreams;s++){
               for (m=1;m<=maxMixes;m++)
                  mixoutp[j][t][s][m]=LZERO;
            }
         }
      }
   }
   if (nStreams>1){
      stroutp = (Vector**)New(&alphaBetaStack, maxT*sizeof(Vector*));
      --stroutp;
      for (t=1;t<=maxT;t++){
         stroutp[t] = (Vector*)New(&alphaBetaStack,(nStates-2)*sizeof(Vector));
         stroutp[t] -= 2;
         for (j=2;j<nStates;j++)
            stroutp[t][j] = CreateVector(&alphaBetaStack,nStreams);
      }
   }
   occr = CreateVector(&gstack,nStates-1);
   zot = CreateVector(&gstack,hset.vecSize);
}

/* ---------------------------- Load Data ------------------------- */


/* CheckData: check data file consistent with HMM definition */
void CheckData(char *fn, BufferInfo info) 
{
   char tpk[80];
   char mpk[80];
   
   if (info.tgtPK != hset.pkind)
      HError(2250,"CheckData: Parameterisation in %s[%s] is incompatible with hmm %s[%s]",
             fn,ParmKind2Str(info.tgtPK,tpk),hmmfn,ParmKind2Str(hset.pkind,mpk));

   if (info.tgtVecSize!=hset.vecSize)
      HError(2250,"CheckData: Vector size in %s[%d] is incompatible with hmm %s[%d]",
             fn,info.tgtVecSize,hmmfn,hset.vecSize);
}

/* InitSegStore : Initialise segStore for particular observation */
void InitSegStore(BufferInfo *info)
{
   Observation obs;
   Boolean eSep;

   SetStreamWidths(info->tgtPK,info->tgtVecSize,hset.swidth,&eSep);
   obs = MakeObservation(&gstack,hset.swidth,info->tgtPK,
                         hset.hsKind==DISCRETEHS,eSep);
   segStore = CreateSegStore(&segmentStack,obs,10);
   firstTime = FALSE;
}

/* LoadFile: load whole file or segments into segStore */
void LoadFile(char *fn)
{
   BufferInfo info;
   char labfn[80];
   Transcription *trans;
   long segStIdx,segEnIdx;
    int segIdx=1;  /* Between call handle on latest seg in segStore */  
    int prevSegIdx=1;
   HTime tStart, tEnd;
   int k,i,s,ncas,nObs,segLen;
   LLink p;
   Observation obs;

   if((pbuf=OpenBuffer(&bufferStack, fn, 10, dff, FALSE_dup, FALSE_dup))==NULL)
      HError(2250,"LoadFile: Config parameters invalid");
   GetBufferInfo(pbuf,&info);
   CheckData(fn,info);
   if (firstTime) InitSegStore(&info);

   if (segId == NULL)  {   /* load whole parameter file */
      nObs = ObsInBuffer(pbuf);
      tStart = 0.0;
      tEnd = (info.tgtSampRate * nObs);
      LoadSegment(segStore, tStart, tEnd, pbuf);
      if (nObs > maxT) 
         maxT=nObs; 
      if (nObs < minT)
         minT=nObs;      
      segIdx++;
   }
   else {                  /* load segment of parameter file */
      MakeFN(fn,labDir,labExt,labfn);
      trans = LOpen(&transStack,labfn,lff);
       
      ncas = CountLabs(trans->head);
      nObs = 0;
      if ( ncas > 0) {
         for (i=1,nObs=0; i<=ncas; i++) {
	   
	    p = GetLabN(trans->head,i);
            segStIdx= (long) (p->start/info.tgtSampRate);
            segEnIdx  = (long) (p->end/info.tgtSampRate);
            
            if (segEnIdx >= ObsInBuffer(pbuf)) 
               segEnIdx = ObsInBuffer(pbuf)-1;
	    /* if (((segEnIdx - segStIdx + 1 >= nStates-2) || !segReject) 
		&& (segStIdx <= segEnIdx)) {	/* skip short segments */
           /* we don't skip  at all
	    */
	   if(1==1){
	   LoadSegment(segStore, p->start, p->end, pbuf);
	        if (trace&T_LD1)
                  printf("  loading seg %s %f[%ld]->%f[%ld]\n",segId->name,
                         p->start,segStIdx,p->end,segEnIdx);
               segLen = SegLength(segStore, segIdx);
               nObs += segLen;
               if (segLen > maxT) 
                  maxT=segLen; 
               if (segLen < minT)
                  minT=segLen;
               segIdx++;
            }else if (trace&T_LD1)
               printf("   seg %s %f->%f ignored\n",segId->name,
                      p->start,p->end);
         }        
      }   
   }
   if (hset.hsKind == DISCRETEHS){
      for (k=prevSegIdx; k<segIdx; k++){
         segLen = SegLength(segStore, k);
         for (i=1; i<=segLen; i++){
            obs = GetSegObs(segStore, k, i);
            for (s=1; s<=nStreams; s++){
               if( (obs.vq[s] < 1) || (obs.vq[s] > maxMixInS[s]))
                  HError(2250,"LoadFile: Discrete data value [ %d ] out of range in stream [ %d ] in file %s",obs.vq[s],s,fn);
            }
         }
      }
      prevSegIdx=segIdx;
   }

   if (trace&T_LD0)
      printf(" %d observations loaded from %s\n",nObs,fn);
   CloseBuffer(pbuf);
   ResetHeap(&transStack);
}


/* ------------------------ Trace Functions -------------------- */

/* ShowSegNum: if not already printed, print seg number */
void ShowSegNum(int seg)
{
   static int lastseg = -1;
   
   if (seg != lastseg){
      printf("---- Training Segment %d [%3d frames] ----\n",seg,T);
      lastseg = seg;
   }
}
   
/* ------------------------- Alpha-Beta ------------------------ */

/* SetOutP: Set the output and mix prob matrices */                        
void SetOutP(int seg)
{
   int i,t,m,mx,s,nMix=0;
   StreamElem *se;
   MixtureElem *me;
   StateInfo *si;
   Matrix mixp;
   LogFloat x,prob,streamP;
   Vector strp = NULL;
   Observation obs;
   TMixRec *tmRec = NULL;
   float wght=0.0,tmp;
   MixPDF *mpdf=NULL;
   PreComp *pMix;
   
   for (t=1;t<=T;t++) {
      obs = GetSegObs(segStore, seg, t);
      if (hsKind == TIEDHS)
         PrecomputeTMix(&hset,&obs,tMPruneThresh,0);         
      if ((maxMixes>1) && (hsKind!=DISCRETEHS)){ /* Multiple Mix Case */
         for (i=2;i<nStates;i++) {
            prob = 0.0;
            si = hmm->svec[i].info;
            se = si->pdf+1; 
            mixp = mixoutp[i][t];
            if (nStreams>1) strp = stroutp[t][i];
            for (s=1;s<=nStreams;s++,se++){
               switch (hsKind){         /* Get nMix */
               case TIEDHS:
                  tmRec = &(hset.tmRecs[s]);
                  nMix = tmRec->nMix;
                  break;
               case PLAINHS:
               case SHAREDHS:
                  nMix = se->nMix;
                  break;
               }
               streamP = LZERO;
               for (mx=1;mx<=nMix;mx++) {
                  m=(hsKind==TIEDHS)?tmRec->probs[mx].index:mx;
                  switch (hsKind){      /* Get wght and mpdf */
                  case TIEDHS:
                     wght=se->spdf.tpdf[m];
                     mpdf=tmRec->mixes[m];
                     break;
                  case PLAINHS:
                  case SHAREDHS:
                     me = se->spdf.cpdf+m;
                     wght=me->weight;
                     mpdf=me->mpdf;
                     break;
                  }
                  if (wght>MINMIX){
                     switch(hsKind) { /* Get mixture prob */
                     case TIEDHS:
                        tmp = tmRec->probs[mx].prob;
                        x = (tmp>=MINLARG)?log(tmp)+tmRec->maxP:LZERO;
                        break;
                     case SHAREDHS : 
                        pMix = (PreComp *)mpdf->hook;
                        if (pMix->time==t)
                           x = pMix->prob;
                        else {
                           x = MOutP(obs.fv[s],mpdf);
                           pMix->prob = x; pMix->time = t;
                        }
                        break;
                     case PLAINHS : 
                        x=MOutP(obs.fv[s],mpdf);
                        break;
                     default:
                        x=LZERO;
                        break;
                     }
                     mixp[s][m]=x;
                     streamP = LAdd(streamP,log(wght)+x);
                  } else
                     mixp[s][m]=LZERO;
               }               
               if (nStreams>1)
                  strp[s]=streamP;
               prob += streamP; /* note stream weights ignored */
            }   
            outprob[i][t]=prob;
         }
      } else 
         if (nStreams>1) {      /* Single Mixture multiple stream */
            for (i=2;i<nStates;i++) {
               prob = 0.0;
               si = hmm->svec[i].info;
               se = si->pdf+1;
               strp = stroutp[t][i];
               for (s=1;s<=nStreams;s++,se++){
                  streamP = SOutP(&hset,s,&obs,se);
                  strp[s] = streamP;
                  prob += streamP; /* note stream weights ignored */
               }
               outprob[i][t]=prob;
            }
         } else                 /* Single Mixture - Single Stream */
            for (i=2;i<nStates;i++){
               si = hmm->svec[i].info;
               se = si->pdf+1;
               if (hsKind==DISCRETEHS)
                  outprob[i][t]=SOutP(&hset,1,&obs,se);
               else
                  outprob[i][t]=OutP(&obs,hmm,i);
            }
   }
   if (trace  & T_OTP) {
      ShowSegNum(seg);
      ShowMatrix("OutProb",outprob,10,12);
   }
}

/* SetAlpha: compute alpha matrix and return prob of given sequence */
LogDouble SetAlpha(int seg)
{
   int i,j,t;
   LogDouble x,a;

   alpha[1][1] = 0.0;
   for (j=2;j<nStates;j++) {              /* col 1 from entry state */
      a=hmm->transP[1][j];
      if (a<LSMALL)
         alpha[j][1] = LZERO;
      else
         alpha[j][1] = a+outprob[j][1];
   }
   alpha[nStates][1] = LZERO;
   
   for (t=2;t<=T;t++) {             /* cols 2 to T */
      for (j=2;j<nStates;j++) {
         x=LZERO ;
         for (i=2;i<nStates;i++) {
            a=hmm->transP[i][j];
            if (a>LSMALL)
               x = LAdd(x,alpha[i][t-1]+a);
         }
	 
         alpha[j][t]=x+outprob[j][t];
      }
      alpha[1][t] = alpha[nStates][t] = LZERO;
   }
   x = LZERO ;                      /* finally calc seg prob */
   for (i=2;i<nStates;i++) {
      a=hmm->transP[i][nStates];
      if (a>LSMALL)
         x=LAdd(x,alpha[i][T]+a); 
   }  
   alpha[nStates][T] = x;
   
   if (trace  & T_ALF) {
      ShowSegNum(seg);
      ShowDMatrix("Alpha",alpha,12,12); 
      printf("LogP= %10.3f\n\n",x);
   }
   return x;
}

/* SetBeta: compute beta matrix */
LogDouble SetBeta(int seg)
{
   int i,j,t;
   LogDouble x,a;

   beta[nStates][T] = 0.0;
   for (i=2;i<nStates;i++)                /* Col T from exit state */
      beta[i][T]=hmm->transP[i][nStates];
   beta[1][T] = LZERO;
   for (t=T-1;t>=1;t--) {           /* Col t from col t+1 */
      for (i=1;i<=nStates;i++)
         beta[i][t]=LZERO ;
      for (j=2;j<nStates;j++) {
         x=outprob[j][t+1]+beta[j][t+1];
         for (i=2;i<nStates;i++) {
            a=hmm->transP[i][j];
            if (a>LSMALL)
               beta[i][t]=LAdd(beta[i][t],x+a);
         }
      }
   }
   x=LZERO ;
   for (j=2;j<nStates;j++) {
      a=hmm->transP[1][j];
      if (a>LSMALL)
         x=LAdd(x,beta[j][1]+a+outprob[j][1]); 
   }
   beta[1][1] = x;
   if (trace & T_BET) {
      ShowSegNum(seg);
      ShowDMatrix("Beta",beta,10,12); 
      printf("LogP=%10.3f\n\n",beta[1][1]);
   }
   return x;
}

/* --------------------- Record Statistics ---------------- */

/* SetOccr: set the global occupation counters occr for current seg */
void SetOccr(LogDouble pr, int seg)
{
   int i,t;
   DVector alpha_i,beta_i;
   Vector a_i;
   LogDouble x;
   
   occr[1] = 1.0;
   for (i=2;i<nStates;i++) {
      alpha_i = alpha[i]; beta_i = beta[i];
      a_i = hmm->transP[i];
      x=LZERO ;
      for (t=1;t<=T;t++)
         x=LAdd(x,alpha_i[t]+beta_i[t]);
      x -= pr;
      if (x>MINEARG) 
         occr[i] = exp(x);
      else
         occr[i] = 0.0;
   }
   if (trace & T_OCC){
      ShowSegNum(seg);
      ShowVector("OCC: ",occr,20);
   }
}

/* FloorMixes: apply floor to given mix set */
void FloorMixes(MixtureElem *mixes, int M, float floor)
{
   float sum,fsum,scale;
   MixtureElem *me;
   int m;
   
   sum = fsum = 0.0;
   for (m=1,me=mixes; m<=M; m++,me++) {
      if (me->weight>floor)
         sum += me->weight;
      else {
         fsum += floor; me->weight = floor;
      }
   }
   if (fsum>1.0)
      HError(2223,"FloorMixes: Floor sum too large");
   scale = (1.0-fsum)/sum;
   if (trace&T_WRE) printf("MIXW: ");
   for (m=1,me=mixes; m<=M; m++,me++){
      if (me->weight>floor)
         me->weight *= scale;
      if (trace&T_WRE) printf(" %.2f",me->weight);
   }
   if (trace&T_WRE) printf("\n");
}  

/* FloorTMMixes: apply floor to given tied mix set */
void FloorTMMixes(Vector mixes, int M, float floor)
{
   float sum,fsum,scale,fltWt;
   int m;
   
   sum = fsum = 0.0;
   for (m=1; m<=M; m++) {
      fltWt = mixes[m];
      if (fltWt>floor)
         sum += fltWt;
      else {
         fsum += floor;
         mixes[m] = floor;
      }
   }
   if (fsum>1.0) HError(2223,"FloorTMMixes: Floor sum too large");
   scale = (1.0-fsum)/sum;
   if (trace&T_WRE) printf("MIXW: ");
   for (m=1; m<=M; m++){
      fltWt = mixes[m];
      if (fltWt>floor)
         mixes[m] = fltWt*scale;
      if (trace&T_WRE) printf(" %.2f",fltWt);
   }
}

/* FloorDProbs: apply floor to given discrete prob set */
void FloorDProbs(ShortVec mixes, int M, float floor)
{
   float sum,fsum,scale,fltWt;
   int m;
   
   sum = fsum = 0.0;
   for (m=1; m<=M; m++) {
      fltWt = Short2DProb(mixes[m]);
      if (fltWt>floor)
         sum += fltWt;
      else {
         fsum += floor;
         mixes[m] = DProb2Short(floor);
      }
   }
   if (fsum>1.0) HError(2327,"FloorDProbs: Floor sum too large");
   if (fsum == 0.0) return;
   if (sum == 0.0) HError(2328,"FloorDProbs: No probabilities above floor");
   scale = (1.0-fsum)/sum;
   for (m=1; m<=M; m++){
      fltWt = Short2DProb(mixes[m]);
      if (fltWt>floor)
         mixes[m] = DProb2Short(fltWt*scale);
   }
}


/* ------------------------- Top Level Control ----------------------- */


char** str_split(char* a_str, const char a_delim)
{
  char** result    = 0;
  size_t count     = 0;
  char* tmp        = a_str;
  char* last_comma = 0;
  char delim[2];
  delim[0] = a_delim;
  delim[1] = 0;

  /* Count how many elements will be extracted. */
  while (*tmp)
    {
      if (a_delim == *tmp)
        {
	  count++;
	  last_comma = tmp;
        }
      tmp++;
    }

  /* Add space for trailing token. */
  count += last_comma < (a_str + strlen(a_str) - 1);

  /* Add space for terminating null string so caller
     knows where the list of returned strings ends. */
  count++;

  result = malloc(sizeof(char*) * count);

  if (result)
    {
      size_t idx  = 0;
      char* token = strtok(a_str, delim);

      while (token)
        {
	  assert(idx < count);
	  *(result + idx++) = strdup(token);
	  token = strtok(0, delim);
        }
      assert(idx == count - 1);
      *(result + idx) = 0;
    }

  return result;
}

/*Compute log likelihood */
void compute_LL(FILE *fp,char *inpLab)
{
   LogFloat segProb,oldP,newP,delta;
   LogDouble ap,bp;
   int converged,iteration,seg;
   
   char * line = (char*)malloc(1000);
   size_t len = 0;
   ssize_t read;
   char **parts;
  
   FILE *fi = fopen(inpLab,"r");
   iteration=0; 
   oldP=LZERO;
   ZeroAccs(&hset, uFlags); newP = 0.0; ++iteration;
   nTokUsed = 0;
   for (seg=1;seg<=nSeg;seg++) {
	read = getline(&line, &len, fi);
       	size_t ln = strlen(line) - 1;
	if (line[ln] == '\n')
	  line[ln] = '\0';
	parts = str_split(line,' ');
	T=SegLength(segStore,seg);
         SetOutP(seg);
         /*if ((ap=SetAlpha(seg)) > LSMALL){*/
	 if (1==1){ /* above line cause some examples to  be escaped */
            bp = SetBeta(seg);
	    ap=SetAlpha(seg);
            if (trace & T_LGP)
               printf("%d.  Pa = %e, Pb = %e, Diff = %e\n",seg,ap,bp,ap-bp);
            segProb = (ap + bp) / 2.0;  /* reduce numeric error */
	    segProb = ap;
	    fprintf(fp,"%s\t%s\t%s\t%f\n",*parts,*(parts+1),segLab,segProb);
            
         } else
            if (trace&T_TOP) 
               printf("Example %d skipped\n",seg);
      }
      free(line);
      if (parts)
	{
	  int i;
	  for (i = 0; *(parts + i); i++)
	    {
	      
	      free(*(parts + i));
	    }
	 
	  free(parts);
	}
      fclose(fi);
}



/* ----------------------------------------------------------- */
/*                      END:  HLlike.c                          */
/* ----------------------------------------------------------- */
