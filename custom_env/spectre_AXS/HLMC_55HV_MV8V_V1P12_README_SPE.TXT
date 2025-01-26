*** HLMC SPICE MODEL FOR 0.055UM LOGIC SALICIDE HIGH-VOLTAGE (1.2V/8.0V/32V) PROCESS ***
*** FOR SPECTRE ONLY ***

*  **********************************************
*  *             HLMC SPICE MODEL               *
*  **********************************************
*    PROCESS :  55NM HV PROCESS (1.2V/8.0V/32V) 
*    MODEL   :  BSIM4
*    VERSION :  V1.12
*    DATE    :  DEC 13, 2023
*    Spectre VERSION : 15.1.0.801.isr17 LATER
************************************************************************************
1. UPDATE HISTORY:

      V1.0(2023-07-10) : INITIATE.
      V1.1(2023-10-20) : 1. ADD BJT/VAR MC CORNER.
                         2. OPEN MV MOS LOD/WPE INSTANCE.
                         3. DELETE NW RELATED DIODE.
                         4. DELETE ALL PNP BJT.
                         5. RENAME PWDNWDIO8 to PDNWDIO8.
                         6. UPDATE ALL LV/HV MOS DEVICE BASE ON 0422'S DATA.
                         7. UPDATE SOME DEVICE'S INSTANCE FOR UNIFORMITY.
                         8. COPY HV_SBD DIODE FROM 6V PLATFORM.
      V1.12(2023-12-13) : 1. UPDATED RES_PPO MODEL BASED ON AE04459#7 AND FINE TUNE RSH IN RES_PPOSAB MODEL.
			              2. UPDATED LVBJT MODEL BASED ON AE04605#2

2. FILES:

      HLMC_55HV_MV8V_V1P12_README_SPE.TXT              .... THIS FILE!
      HLMC_55HV_MV8V_V1P12.LIB                         .... MAIN MODEL FOR MOSFETS,DIODE,BJT,MOM CAPACITORS AND RESISTORS
      HLMC_55HV_MV8V_V1P12_CORNER.LIB                  .... DEFAULT MODEL CORNER COMBINATION FILE
                                           
3. HOW TO USE HLMC SPICE MODELS IN Spectre:

      A. FOR CORNER SIMULATION (NO MISMATCH), PLEASE ADD THE FOLLOWING STATEMENTS TO Spectre NETLIST.
      SPECIFY MOS, DIODE, BJT, RESISTOR, MOM,SBD DIODE AND MOS VARACTOR BY THE '.LIB' STATEMENT
        .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' TT
        .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' DIO_ALL_TT
        .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' BJT_ALL_TT
        .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' RES_ALL_TT
        .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' VAR_ALL_TT
        .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' MOM_ALL_TT
      WHERE '/XXX/XXX/' IS THE DIRECTORY WHERE LIB FILES IS LOCATED. 

      B. FOR STATISTICAL MODEL SIMULATION, PLEASE ADD THE FOLLOWING STATEMENTS TO NETLIST.
      I. FOR 1.2V/32V MOSFETS MODEL, SPECIFY MOS BY THE '.LIB' STATEMENT FOR GLOBAL VARIATION AND BY SETTING FLAG MISMOD=1 OR MISMOD=0 TO TURN ON OR TURN OFF LOCAL VARIATION (MISMATCH MODEL)
         FOR 8V MOSFETS MODEL, SPECIFY MOS BY THE '.LIB' STATEMENT FOR GLOBAL VARIATION AND BY SETTING FLAG MISMATCH=1 OR MISMATCH=0 TO TURN ON OR TURN OFF LOCAL VARIATION (MISMATCH MODEL)
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' TT_GLOBAL
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' FF_GLOBAL
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' SS_GLOBAL
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' FS_GLOBAL
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' SF_GLOBAL
		 
      II. FOR 1.2V/32V MOSFETS MODEL, SPECIFY MOS BY THE '.LIB' STATEMENT FOR MONTE CARLO SIMULATION AND BY SETTING FLAG MISMOD=1 OR MISMOD=0 TO TURN ON OR TURN OFF LOCAL VARIATION (MISMATCH MODEL)
          FOR 8V MOSFETS MODEL, SPECIFY MOS BY THE '.LIB' STATEMENT FOR MONTE CARLO SIMULATION AND BY SETTING FLAG MISMATCH=1 OR MISMATCH=0 TO TURN ON OR TURN OFF LOCAL VARIATION (MISMATCH MODEL)
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' MC
		 
      III. SPECIFY PASSIVE DEVICE BY THE '.LIB' STATEMENT BY SETTING FLAG MISMOD=1 OR MISMOD=0(MOM_MISMATCH IS FOR MOM) TO TURN ON OR TURN OFF LOCAL VARIATION (MISMATCH MODEL)
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' BJT_ALL_TT
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' RES_ALL_TT
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' VAR_ALL_TT
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' MOM_ALL_TT
		 
      IIII. SPECIFY PASSIVE DEVICE BY THE '.LIB' STATEMENT FOR MONTE CARLO MODEL AND BY SETTING FLAG MISMOD=1 OR MISMOD=0(MOM_MISMATCH IS FOR MOM) TO TURN ON OR TURN OFF LOCAL VARIATION (MISMATCH MODEL)
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' BJT_ALL_MC
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' RES_ALL_MC
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' VAR_ALL_MC
         .LIB '/XXX/XXX/HLMC_55HV_MV8V_V1P12_CORNER.LIB' MOM_ALL_MC
         
      WHERE '/XXX/XXX/' IS THE DIRECTORY WHERE LIB FILES IS LOCATED. 
        
4. THE CAPABILITY OF MODEL

      A. MOS MODEL    
        
        *--------------------------------------------------------* 
        |     MOSFET TYPE    |    NAME   |    WMIN   |    LMIN   |
        |========================================================| 
        |      1.2V NMOS     |    N12    |  0.108UM  |  0.054UM  |
        *--------------------------------------------------------*
        |      1.2V PMOS     |    P12    |  0.108UM  |  0.054UM  |
        *--------------------------------------------------------*
        |    1.2V LVT NMOS   |  NLVT12   |  0.108UM  |  0.054UM  |
        *--------------------------------------------------------*
        |    1.2V LVT PMOS   |  PLVT12   |  0.108UM  |  0.054UM  |
        *--------------------------------------------------------*
        | 32V SYMMETRIC NMOS |   N32S    |   4.0UM   |   2.1UM   |
        *--------------------------------------------------------*
        | 32V SYMMETRIC PMOS |   P32S    |   4.0UM   |   2.1UM   |
        *--------------------------------------------------------*
        | 32V ASYMMETRIC NMOS|   N32A    |   4.0UM   |   2.3UM   |
        *--------------------------------------------------------*
        | 32V ASYMMETRIC PMOS|   P32A    |   4.0UM   |   2.3UM   |
        *--------------------------------------------------------*
        |      8.0V NMOS     |   N8IO    |   0.3UM   |   0.9UM   |
        *--------------------------------------------------------*
        |      8.0V PMOS     |   P8IO    |   0.3UM   |   0.8UM   |
        *--------------------------------------------------------*
        |  8.0V NMOS UD 6V   |  N8UD6IO  |   0.3UM   |   0.8UM   |
        *--------------------------------------------------------*
        |  8.0V PMOS UD 6V   |  P8UD6IO  |   0.3UM   |   0.7UM   |
        *--------------------------------------------------------*
        |  8.0V NMOS UD 5V   |  N8UD5IO  |   0.3UM   |   0.8UM   |
        *--------------------------------------------------------*
        |  8.0V PMOS UD 5V   |  P8UD5IO  |   0.3UM   |   0.7UM   |
        *--------------------------------------------------------*
        |  8.0V NMOS UD 3.3V |  N8UD33IO |   0.3UM   |   0.8UM   |
        *--------------------------------------------------------*
        |  8.0V PMOS UD 3.3V |  P8UD33IO |   0.3UM   |   0.7UM   |
        *--------------------------------------------------------*
 
  	  TEMPERATURE RANGE:-40C~125C

      B. BJT MODEL 
        *---------------------------------------------------------* 
        |      BJT TYPE         |     NAME        |  EMITTER AREA |
        |=========================================================|
        |   1.2V NPN_13.5X13.5  |  NPN12A182_CKT  |13.5*13.5 UM^2 |
        *---------------------------------------------------------*  
        |   1.2V NPN_9X9        |  NPN12A81_CKT   |   9*9 UM^2    |
        *---------------------------------------------------------*  
        |   1.2V NPN_4.5X4.5    |  NPN12A20_CKT   | 4.5*4.5 UM^2  |
        *---------------------------------------------------------* 
        |   1.2V NPN_1.8X1.8    |  NPN12A3P2_CKT  | 1.8*1.8 UM^2  | 
        |---------------------------------------------------------|
        |   8.0V NPN_10X10      |  NPN80A100_CKT  |   10*10 UM^2  |
        *---------------------------------------------------------*  
        |   8.0V NPN_5X5        |  NPN80A25_CKT   |  5.0*5.0 UM^2 |
        *---------------------------------------------------------* 
        |   8.0V NPN_2X2        |  NPN80A4_CKT    |  2.0*2.0 UM^2 |
        *---------------------------------------------------------* 

      TEMPERATURE RANGE:-40C~125C

      C. JUNCTION DIODE MODEL 
        *---------------------------------------*
        |      DIODE TYPE          |   NAME     |
        |=======================================|
        |     1.2V N+/PWELL        | NDIO12     |
        *---------------------------------------*
        |     1.2V P+/NWELL        | PDIO12     |
        *---------------------------------------*
        |     1.2V LVT N+/PWELL    | LVTNDIO12  |
        *---------------------------------------*
        |     1.2V LVT P+/NWELL    | LVTPDIO12  |
        *---------------------------------------*
        |     1.2V DNWELL/PWELL    | PDNWDIO12  | 
        *---------------------------------------*
        |     32V NDRIFT/HVPWELL   | NDRPWDIO32 |
        *---------------------------------------*
        |     32V PDRIFT/HVNWELL   | PDRNWDIO32 |
        *---------------------------------------*
        |     1.2V SBD DIODE       | SBD_LV     |
        *---------------------------------------*
        |     HV SBD DIODE         | SBD_HV     |
        *---------------------------------------*
        |     8V N+/PWELL          | NDIO8      |
        *---------------------------------------*
        |     8V P+/NWELL          | PDIO8      |
        *---------------------------------------*  
        |     8V PWELL/DNWELL      | PDNWDIO8   | 
        *---------------------------------------*
        |     DNWELL/PSUB          | DNWPSUB    |
        *---------------------------------------*  
      TEMPERATURE RANGE:-40C~125C

      D. RESISTOR MODEL
        *-----------------------------------------------------------*  
        |       RESISTOR TYPE                       |      NAME     | 
        |===========================================================|  
        | SILICIDE N+ POLY                          |    RES_NPO    | 
        |-----------------------------------------------------------|  
        | SILICIDE P+ POLY                          |    RES_PPO    | 
        |-----------------------------------------------------------| 
        | NON-SILICIDE N+ DIFFUSION                 |  RES_NDIFSAB  | 
        |-----------------------------------------------------------| 
        | NON-SILICIDE P+ DIFFUSION                 |  RES_PDIFSAB  | 
        |-----------------------------------------------------------| 
        | NON-SILICIDE N+ POLY                      |  RES_NPOSAB   | 
        |-----------------------------------------------------------| 
        | NON-SILICIDE P+ POLY                      |  RES_PPOSAB   |
        |-----------------------------------------------------------| 
        | NON-SILICIDE HV N+ POLY                   | RES_NPOSAB_HV | 
        |-----------------------------------------------------------| 
        | NON-SILICIDE HV P+ POLY                   | RES_PPOSAB_HV |
        |-----------------------------------------------------------| 
        | NON-SILICIDE 1K HIGH RESISTOR POLY        |   RES_HRP1K   | 
        |-----------------------------------------------------------| 
        | NON-SILICIDE 2K HIGH RESISTOR POLY        |   RES_HRP2K   |
        |-----------------------------------------------------------| 
        | METAL 1                                   |    RES_M1     | 
        |-----------------------------------------------------------| 
        | METAL 2                                   |    RES_M2     | 
        |-----------------------------------------------------------| 
        | METAL 3                                   |    RES_M3     | 
        |-----------------------------------------------------------| 
        | METAL 4                                   |    RES_M4     | 
        |-----------------------------------------------------------|
        | METAL 5                                   |    RES_M5     | 
        |-----------------------------------------------------------| 
        | METAL 6                                   |    RES_M6     | 
        |-----------------------------------------------------------| 
        | METAL 7                                   |    RES_M7     | 
        |-----------------------------------------------------------| 
        | TOP METAL 1(2X)                           |    RES_T2M1   |
        |-----------------------------------------------------------| 
        | TOP METAL 1(4X)                           |    RES_T4M1   |
        |-----------------------------------------------------------| 
        | TOP METAL 2(4X)                           |    RES_T4M2   | 
        |-----------------------------------------------------------| 
        | ALPA METAL                                |    RES_ALPA   |
        *-----------------------------------------------------------*  
      TEMPERATURE RANGE:-40C~125C

      E.MOM MODEL 
        *---------------------------------------------------------------------------------------------------------------------*
        |                                                      MOM CAPACITOR                                                  |
        *---------------------------------------------------------------------------------------------------------------------*
        |                  |             |              DEFAULT            |               MIN.              |                | 
        |=====================================================================================================================|
        |   MOM TYPE       |     NAME    |  FINGER NUM. |  FINGER LENGTH   |  FINGER NUM  |  FINGER LENGTH   | FIXED SPC&LEN  | 
        |=====================================================================================================================|
        |       M1-M2      |     MOM12   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M1-M3      |     MOM13   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M1-M4      |     MOM14   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M1-M5      |     MOM15   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M1-M6      |     MOM16   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M2-M3      |     MOM23   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M2-M4      |     MOM24   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M2-M5      |     MOM25   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M2-M6      |     MOM26   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M3-M4      |     MOM34   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M3-M5      |     MOM35   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M3-M6      |     MOM36   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M4-M5      |     MOM45   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M4-M6      |     MOM46   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        |       M5-M6      |     MOM56   |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M1-M2 3-TERMINAL |   MOM12_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M1-M3 3-TERMINAL |   MOM13_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M1-M4 3-TERMINAL |   MOM14_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M1-M5 3-TERMINAL |   MOM15_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M1-M6 3-TERMINAL |   MOM16_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M2-M3 3-TERMINAL |   MOM23_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M2-M4 3-TERMINAL |   MOM24_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M2-M5 3-TERMINAL |   MOM25_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M2-M6 3-TERMINAL |   MOM26_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M3-M4 3-TERMINAL |   MOM34_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M3-M5 3-TERMINAL |   MOM35_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M3-M6 3-TERMINAL |   MOM36_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M4-M5 3-TERMINAL |   MOM45_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M4-M6 3-TERMINAL |   MOM46_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
        | M5-M6 3-TERMINAL |   MOM56_3T  |      10      |        5UM       |       2      |       0.9UM      |     0.09UM     |
        *---------------------------------------------------------------------------------------------------------------------*
      LR: METAL LENGTH IN UNIT
      NF: METAL FINGER IN UNIT
      MR: NUMBER OF UNIT
      TEMPERATURE RANGE:-40C~125C

      F. MOS VARACTOR MODEL 
       *-------------------------------------------------------*
       |    MOS VARACTOR TYPE                     |    NAME    |
       |=======================================================| 
       |      1.2V N+POLY/NWELL MOS VARACTOR      | NVAR12_CKT |
       |-------------------------------------------------------|
       |      1.2V P+POLY/PWELL MOS VARACTOR      | PVAR12_CKT |
       *-------------------------------------------------------*
       |      8V N+POLY/NWELL MOS VARACTOR        | NVAR8_CKT  |
       *-------------------------------------------------------*

5. CORNER MODEL 

      ELEVEN MODEL CORNERS ARE PROVIDED FOR MOSFETS, 
      THREE MODEL CORNERS ARE PROVIDED FOR DIODE,
      THREE MODEL CORNERS ARE PROVIDED FOR BJT.
      FOUR MODEL CORNERS ARE PROVIDED FOR RES.
      THREE MODEL CORNERS ARE PROVIDED FOR VARACTORS.
      FOUR MODEL CORNERS ARE PROVIDED FOR MOM.

      THEY ARE
        
      ----------------------------------------------------
      MOS        NAME : CORNER
      ----------------------------------------------------
                 TT : TYPICAL CASE
                 SS : SLOW CASE
                 FF : FAST CASE
                 SF : SLOW N FAST P CASE    
                 FS : FAST N SLOW P CASE    
                 TT_GLOBAL : TYPICAL CASE OF GLOBAL VARIATION
                 SS_GLOBAL : SLOW CASE OF GLOBAL VARIATION
                 FF_GLOBAL : FAST CASE OF GLOBAL VARIATION
                 SF_GLOBAL : SLOW N FAST P CASE OF GLOBAL VARIATION    
                 FS_GLOBAL : FAST N SLOW P CASE OF GLOBAL VARIATION    
                 MC : MONTE CARLO MODEL     
      ----------------------------------------------------
      DIODE      NAME : CORNER
      ----------------------------------------------------
                 DIO_ALL_TT : TYPICAL CASE
                 DIO_ALL_SS : SLOW CASE
                 DIO_ALL_FF : FAST CASE  
      ----------------------------------------------------
      BJT        NAME : CORNER
      ----------------------------------------------------
                 BJT_ALL_TT : TYPICAL CASE
                 BJT_ALL_SS : SLOW CASE
                 BJT_ALL_FF : FAST CASE 
      ----------------------------------------------------
      RES        NAME : CORNER
      ----------------------------------------------------
                 RES_ALL_TT : TYPICAL CASE
                 RES_ALL_SS : SLOW CASE
                 RES_ALL_FF : FAST CASE 
                 RES_ALL_MC : MONTE CARLO MODEL 
      ----------------------------------------------------
      MOM        NAME : CORNER
      ----------------------------------------------------
                 MOM_ALL_TT : TYPICAL CASE
                 MOM_ALL_SS : SLOW CASE
                 MOM_ALL_FF : FAST CASE 
                 MOM_ALL_MC : MONTE CARLO MODEL
      ----------------------------------------------------
      VARACTOR   NAME : CORNER
      ----------------------------------------------------
                 VAR_ALL_TT : TYPICAL CASE
                 VAR_ALL_SS : SLOW CASE
                 VAR_ALL_FF : FAST CASE 


 
6. EXAMPLE NETLIST FOR SPICE MODEL
   
--------------------------------------------------------------
*Spectre NETLIST FOR MOS
.OPTIONS  ITL5=0 MEASDGT=5 POST=2 NOMOD INGOLD=2
.LIB 'HLMC_55HV_MV8V_V1P12_CORNER.LIB' TT
.TEMP 25 
.PARAM VDD=1.2 VDD1=-1.2

M2 MD2 MG2 MS2 MB2 N12 W=WR L=LR SA=0.4U SB=0.4U
VMD2 MD2 0 VDD
VMG2 MG2 0 VDD
VMS2 MS2 0 0
VMB2 MB2 0 0
.DC VMD2 0 VDD VDD SWEEP DATA = VDOT
.MEAS DC IDSAT_N12 FIND I(VMD2) WHEN V(MD2,0) =VDD

M20 MD20 MG20 MS20 MB20 P12  W=WR L=LR SA=0.4U SB=0.4U
VMD20 MD20 0 VDD1
VMG20 MG20 0 VDD1
VMS20 MS20 0 0
VMB20 MB20 0 0
.DC VMD20 0 VDD1 VDD1 SWEEP DATA = VDOT
.MEAS DC IDSAT_P12 FIND I(VMD20) WHEN V(MD20,0) =VDD1

.PARAM
.DC DATA=VDOT
.DATA VDOT
 WR     LR
0.12U 0.06U
0.12U 10U
0.3U 0.06U
0.6U 0.06U
10U 0.06U
10U 10U
.ENDDATA
.END


--------------------------------------------------------------
*Spectre NETLIST FOR MOS MISMATCH

.OPTION  LIST POST=2
VDS 30 0 VDD
VGS 20 0 5
VGS2 21 20 -0.01
VBS 40 0 VBB

X1 30 20 0 40 N5IO_MAC W=10U L=0.6U
.PARA VDD=0.05
.PARA VBB=0
.TEMP 25
.LIB 'HLMC_55HV_MV8V_V1P12_CORNER.LIB' TT_GLOBAL

.DC VGS 0 5.0 0.01 
+MONTE=100
.PLOT I(X1.M1)
.MEASURE DC IDLIN FIND I(X1.M1) WHEN PAR('V(20)')=5
.END

---------------------------------------------------------------

---------------------------------------------------------------
*Spectre NETLIST FOR DIODE
.OPTION DCCAP 
.PROTECT
.LIB 'HLMC_55HV_MV8V_V1P12_CORNER.LIB' DIO_ALL_TT
.UNPROTECT
.TEMP 25
D1 1 0 NDIO12 AREA = 9.00E-012 PJ = 1.20E-05
D2 2 0 PDIO12 AREA = 9.00E-012 PJ = 1.20E-05
V1 1 0 -1
V2 2 0 -1
.OPTION POST
.OP
.END
---------------------------------------------------------------

---------------------------------------------------------------
*Spectre NETLIST FOR SBD DIODE
.OPTION DCCAP 
.PROTECT
.LIB 'HLMC_55HV_MV8V_V1P12_CORNER.LIB' DIO_ALL_TT
.UNPROTECT
.TEMP 25
XD1 1 0 SBD_LV AREA = 9.00E-012 PJ = 1.20E-05
XD2 2 0 SBD_HV AREA = 9.00E-012 PJ = 1.20E-05
V1 1 0 -1
V2 2 0 -1
.OPTION POST
.OP
.END

---------------------------------------------------------------

---------------------------------------------------------------
*Spectre NETLIST FOR BJT
.OPTIONS ITL5=0 MEASDGT=5 POST=2 NOMOD INGOLD=2 CAPTAB
.LIB 'HLMC_55HV_MV8V_V1P12_CORNER.LIB' BJT_ALL_TT
.TEMP 25
VC VC 0 -1
VB VB 0 -1
XQ1 VC VB 0 PNP12A100_CKT
XQ2 VC VB 0 PNP12A25_CKT
XQ3 VC VB 0 PNP12A4_CKT
.END
---------------------------------------------------------------

---------------------------------------------------------------
*Spectre NETLIST FOR MOM CAPACITOR
---------------------------------------------------------------

$QA SIMULATION
.OPTIONS ITL5=0 MEASDGT=5 POST=2 NOMOD INGOLD=2 CAPTAB 
.LIB 'HLMC_55HV_MV8V_V1P12_CORNER.LIB' MOM_ALL_TT
.TEMP 25 125 -40

******************************* MOM CAPACITANCE ****************************

XMOM12    MOM12    GND     MOM12     LR=5E-6 NF=10 MR=1
XMOM12_3T MOM12_3T GND SUB MOM12_3T  LR=5E-6 NF=10 MR=1

XMOM13    MOM13    GND     MOM13     LR=5E-6 NF=10 MR=1
XMOM13_3T MOM13_3T GND SUB MOM13_3T  LR=5E-6 NF=10 MR=1

XMOM14    MOM14    GND     MOM14     LR=5E-6 NF=10 MR=1
XMOM14_3T MOM14_3T GND SUB MOM14_3T  LR=5E-6 NF=10 MR=1

XMOM15    MOM15    GND     MOM15     LR=5E-6 NF=10 MR=1
XMOM15_3T MOM15_3T GND SUB MOM15_3T  LR=5E-6 NF=10 MR=1

XMOM16    MOM16    GND     MOM16     LR=5E-6 NF=10 MR=1
XMOM16_3T MOM16_3T GND SUB MOM16_3T  LR=5E-6 NF=10 MR=1

XMOM23    MOM23    GND     MOM23     LR=5E-6 NF=10 MR=1
XMOM23_3T MOM23_3T GND SUB MOM23_3T  LR=5E-6 NF=10 MR=1

XMOM24    MOM24    GND     MOM24     LR=5E-6 NF=10 MR=1
XMOM24_3T MOM24_3T GND SUB MOM24_3T  LR=5E-6 NF=10 MR=1

XMOM25    MOM25    GND     MOM25     LR=5E-6 NF=10 MR=1
XMOM25_3T MOM25_3T GND SUB MOM25_3T  LR=5E-6 NF=10 MR=1

XMOM26    MOM26    GND     MOM26     LR=5E-6 NF=10 MR=1
XMOM26_3T MOM26_3T GND SUB MOM26_3T  LR=5E-6 NF=10 MR=1

XMOM34    MOM34    GND     MOM34     LR=5E-6 NF=10 MR=1
XMOM34_3T MOM34_3T GND SUB MOM34_3T  LR=5E-6 NF=10 MR=1

XMOM35    MOM35    GND     MOM35     LR=5E-6 NF=10 MR=1
XMOM35_3T MOM35_3T GND SUB MOM35_3T  LR=5E-6 NF=10 MR=1

XMOM36    MOM36    GND     MOM36     LR=5E-6 NF=10 MR=1
XMOM36_3T MOM36_3T GND SUB MOM36_3T  LR=5E-6 NF=10 MR=1

XMOM45    MOM45    GND     MOM45     LR=5E-6 NF=10 MR=1
XMOM45_3T MOM45_3T GND SUB MOM45_3T  LR=5E-6 NF=10 MR=1

XMOM46    MOM46    GND     MOM46     LR=5E-6 NF=10 MR=1
XMOM46_3T MOM46_3T GND SUB MOM46_3T  LR=5E-6 NF=10 MR=1

XMOM56    MOM56    GND     MOM56     LR=5E-6 NF=10 MR=1
XMOM56_3T MOM56_3T GND SUB MOM56_3T  LR=5E-6 NF=10 MR=1


VMOM      1   0      10
VMOM2     GND 0      0
VMOM3     SUB 0      0

RMOM12    1 MOM12    1E-3
RMOM12_3T 1 MOM12_3T 1E-3

RMOM13    1 MOM13    1E-3
RMOM13_3T 1 MOM13_3T 1E-3

RMOM14    1 MOM14    1E-3
RMOM14_3T 1 MOM14_3T 1E-3

RMOM15    1 MOM15    1E-3
RMOM15_3T 1 MOM15_3T 1E-3

RMOM16    1 MOM16    1E-3
RMOM16_3T 1 MOM16_3T 1E-3

RMOM23    1 MOM23    1E-3
RMOM23_3T 1 MOM23_3T 1E-3

RMOM25    1 MOM25    1E-3
RMOM25_3T 1 MOM25_3T 1E-3

RMOM26    1 MOM26    1E-3
RMOM26_3T 1 MOM26_3T 1E-3

RMOM35    1 MOM35    1E-3
RMOM35_3T 1 MOM35_3T 1E-3

RMOM36    1 MOM36    1E-3
RMOM36_3T 1 MOM36_3T 1E-3

RMOM46    1 MOM46    1E-3
RMOM46_3T 1 MOM46_3T 1E-3

.DC VMOM -10 10 1
.PRINT DC CAP(MOM12) CAP(MOM12_3T) CAP(MOM13) CAP(MOM13_3T) 
.PRINT DC CAP(MOM14) CAP(MOM14_3T) CAP(MOM15) CAP(MOM15_3T) 
.PRINT DC CAP(MOM16) CAP(MOM16_3T) CAP(MOM23) CAP(MOM23_3T) 
.PRINT DC CAP(MOM25) CAP(MOM25_3T) CAP(MOM26) CAP(MOM26_3T) 
.PRINT DC CAP(MOM35) CAP(MOM35_3T) CAP(MOM36) CAP(MOM36_3T) 
.PRINT DC CAP(MOM46) CAP(MOM46_3T) 


.OP
.END
---------------------------------------------------------------
