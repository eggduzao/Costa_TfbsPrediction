
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------
% Masters Dissertation by Eduardo Gade Gusmao, 2012
% Based on:
%     - PhDthesis version 2 style file by Jakob Suckale, 2007
%     - CUEDthesis version 1 by Harish Bhanderi 2002
%---------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Identification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Identification
\ProvidesClass{Latex/Classes/PhDthesisPSnPDF}[2007/09/06 v2 PhD thesis class]

% Book style
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\ifx\pdfoutput\undefined
   \LoadClass[dvips, a4paper]{book}
\else
   \LoadClass[pdftex, a4paper]{book}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Packages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Default version Packages
\usepackage[latin1]{inputenc}
\usepackage{amssymb}
\usepackage{graphics} % for improved inclusion of graphics
\usepackage{graphicx}
\usepackage{epstopdf}
\usepackage[margin=10pt,font=small,labelfont=bf]{caption} % for improved layout of figure captions with extra margin, smaller font than text
\usepackage{fancyhdr} % for better header layout
\usepackage{eucal}
\usepackage[portuguese]{babel}
\usepackage[usenames, dvipsnames]{color}
\usepackage[perpage]{footmisc}
%\usepackage[round, sort, numbers]{natbib}
\usepackage[square, sort]{natbib}
\usepackage{ifthen}
\usepackage{multicol} % for pages with multiple text columns, e.g. References
\setlength{\columnsep}{20pt} % space between columns; default 10pt quite narrow
\usepackage[nottoc]{tocbibind} % correct page numbers for bib in TOC, nottoc suppresses an entry for TOC itself
\usepackage{ifpdf}

% My packages
\usepackage{setspace}
\RequirePackage{filecontents}

\usepackage{amsmath,amsthm}
\usepackage{array}
\usepackage{multirow}
\usepackage{rotating}
\usepackage[section]{placeins}
%\usepackage{hyphenat}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Glossary setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Glossary/Abbrev./Symbols
\usepackage[intoc]{nomencl} % load nomencl extension; include in TOC
%\nomrefpage % to include page numbers after abbrevations
\renewcommand{\nomname}{Glossário} % rename nomenclature
\renewcommand{\nomlabel}[1]{\textbf{#1}} % make abbreviations bold
\makenomenclature % used to be \makeglossary
\newcommand{\g}{\footnote{For all abbreviations see the glossary on page \pageref{nom}.}} % type "\g" to refer to glossary


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% PDF/PS setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ifpdf

    \usepackage[ pdftex, plainpages = false, pdfpagelabels, 
                 pdfpagelayout = useoutlines,
                 bookmarks,
                 bookmarksopen = true,
                 bookmarksnumbered = true,
                 breaklinks = true,
                 linktocpage,
                 pagebackref,
                 colorlinks = false,  % was true
                 linkcolor = blue,
                 urlcolor  = blue,
                 citecolor = red,
                 anchorcolor = green,
                 hyperindex = true,
                 hyperfigures
                 ]{hyperref} 

    \DeclareGraphicsExtensions{.png, .jpg, .jpeg, .pdf, .gif, .eps} %GIF doesn't work??
    %\usepackage[pdftex]{graphicx}
    \pdfcompresslevel=9
    \graphicspath{{0_frontmatter/figures/PNG/}{0_frontmatter/figures/PDF/}{0_frontmatter/figures/}}
\else
    \usepackage[ dvips, 
                 bookmarks,
                 bookmarksopen = true,
                 bookmarksnumbered = true,
                 breaklinks = true,
                 linktocpage,
                 pagebackref,
                 colorlinks = true,
                 linkcolor = blue,
                 urlcolor  = blue,
                 citecolor = red,
                 anchorcolor = green,
                 hyperindex = true,
                 hyperfigures
                 ]{hyperref}

    \DeclareGraphicsExtensions{.eps, .ps}
    \usepackage{epsfig}
    \usepackage{graphicx}
    \graphicspath{{0_FrontMatter/figures/EPS/}{0_FrontMatter/figures/}}
\fi


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Page setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%A4 settings
\ifpdf
   \pdfpageheight=297mm
   \pdfpagewidth=210mm
\else
   \setlength{\paperheight}{297mm}
   \setlength{\paperwidth}{210mm}
\fi

\setlength{\hoffset}{0.00cm}
\setlength{\voffset}{0.00cm}

% Two-sided printing
%\setlength{\oddsidemargin}{1.5cm}
%\setlength{\evensidemargin}{0cm}
%\setlength{\topmargin}{1mm}
%\setlength{\headheight}{1.36cm}
%\setlength{\headsep}{1.00cm}
%\setlength{\textheight}{20.84cm}
%\setlength{\textwidth}{14.5cm}
%\setlength{\marginparsep}{1mm}
%\setlength{\marginparwidth}{3cm}
%\setlength{\footskip}{2.36cm}

% One-sided printing
\setlength{\evensidemargin}{0.0cm}
\setlength{\oddsidemargin}{0.0cm}
\setlength{\topmargin}{1mm}
\setlength{\headheight}{0.36cm}
\setlength{\headsep}{1.00cm}
\setlength{\textheight}{22.20cm}
\setlength{\textwidth}{15.95cm}
\setlength{\marginparsep}{0mm}
\setlength{\marginparwidth}{0cm}
\setlength{\footskip}{2.36cm}

% Fancy page layout options
\pagestyle{fancy}
\fancyhead{}
\fancyfoot{}
\fancyhead[CO,CE]{}
\fancyfoot[C]{Confidential}
%\fancyfoot[RO,LE] {\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}
\fancyhead[RO]{\bfseries\rightmark}
\fancyhead[LE]{\bfseries\leftmark}
\fancyfoot[C]{\thepage}
%\headrulewidth{0.4pt}
%\footrulewidth{0.4pt}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Declarations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These macros are used to declare arguments needed for the
% construction of the title page and other preamble.

% Numbering of first pages to roman
\renewcommand{\thepage}{\roman{page}}

% Submitted text declaration
\newcommand{\submittedtext}{{A thesis submitted for the degree of}}

% Cover declarations
\def\degreedate#1{\gdef\@degreedate{#1}}
\def\degree#1{\gdef\@degree{#1}}
\def\collegeordept#1{\gdef\@collegeordept{#1}}
\def\university#1{\gdef\@university{#1}}
\def\crest#1{\gdef\@crest{#1}}

% Single environment
% These macros define an environment for front matter that is always
% single column even in a double-column document.
\newenvironment{alwayssingle}{%
       \@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
       \else\newpage\fi}
       {\if@restonecol\twocolumn\else\newpage\fi}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Pages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Title page
\renewcommand{\maketitle}{
  \begin{alwayssingle}
    \renewcommand{\footnotesize}{\small}
    \renewcommand{\footnoterule}{\relax}
    \thispagestyle{empty}
    \begin{center}
      {\Huge {\bfseries {\@title}} \par}
      {\large \vspace*{35mm} {{\@crest} \par} \vspace*{35mm}}
      {{\Large \@author} \par}
      {\large
        \vspace*{1ex}
        {{\@collegeordept} \par}
        \vspace*{1ex}
        {{\@university} \par}
        \vspace*{25mm}
        {{\submittedtext} \par}
        \vspace*{1ex}
        {\it {\@degree} \par}
        \vspace*{2ex}
        {\@degreedate}
      }
    \end{center}
    \null\vfill
  \end{alwayssingle}
}

% Second page
\newenvironment{makesecondpage}{
  \begin{alwayssingle}
    \renewcommand{\footnotesize}{\small}
    \renewcommand{\footnoterule}{\relax}
    \thispagestyle{empty}
    \begin{center}
      { \Huge {\bfseries {\@title}} \par}
    \end{center}
    \vspace*{15mm}
    \hspace{6 cm}
    \parbox{10cm}{ \setstretch{1.6} {\Large \emph{ \defaulttext }}}
    \vspace*{15mm}
    \begin{center}
      {{\Large \@author} \par}
      {\large
        \vspace*{1mm}
        {{\@collegeordept} \par}
        \vspace*{1mm}
        {{\@university} \par}
        \vspace*{10mm}
        {{\Large {\advisortext} \par}}
        \vspace*{1mm}
        {{\Large {\advisor} \par}}
        \vspace*{10mm}
        {{\Large {\coadvisortext} \par}}
        \vspace*{1mm}
        {{\Large {\coadvisor} \par}}
        \vspace*{11.5mm}
        %{{\advisortitle} \par}
        %\vspace*{15mm}
        {{\submittedtext} \par}
        \vspace*{1.75mm}
        {\it {\@degree} \par}
	\vspace*{3.75mm}
        {\@degreedate}
      }
    \end{center}
    \null\vfill
  \end{alwayssingle}
}

% Library Requirements
\newenvironment{libreq}{
  \pagestyle{plain}
  \vspace*{10cm}
  \hspace{1 cm}
  \parbox{13cm}{
    \setstretch{1.0}{
      \large
      \libreqname \par
      \vspace*{0.5ex}
      \hspace{7 mm} \libreqtitle \par
      \vspace*{0.5ex}
      \hspace{14 mm} \libreqpages \par
      \vspace*{2ex}
      \hspace{7 mm} \libreqinstitute \par
      \vspace*{2ex}
      \hspace{7 mm} \libreqbib \par
      \vspace*{2ex}
      \hspace{7 mm} \libreqarea \par
      \vspace*{2ex}
      \hspace{7 mm} \libreqcode
    }
  }
}

% Reviewers
\newenvironment{reviewerspage}{
  \pagestyle{plain}
  { \setstretch{1.5}
    { \large
      \reviewersintro \par
      \vspace*{1.5cm}
      Aprovado: \hspace{5 mm} \rule{10 cm}{0.5 pt} \par
      %\hspace{2.3 cm} \reviewersorienter \hspace{0.5 mm} -- \reviewersorienterinst \par
      %\vspace*{-1ex}
      %\hspace{6.5 cm} {\bf Orientador} \par
      %\vspace*{5ex}
      %\hspace{2.6 cm} \rule{10 cm}{0.5 pt} \par
      \hspace{2.5 cm} \reviewersone \hspace{0.5 mm} -- \reviewersoneinst \par
      \vspace*{10ex}
      \hspace{2.6 cm} \rule{10 cm}{0.5 pt} \par
      \hspace{1.8 cm} \reviewerstwo \hspace{0.5 mm} -- \reviewerstwoinst \par
      \vspace*{10ex}
      \hspace{2.6 cm} \rule{10 cm}{0.5 pt} \par
      \hspace{1.6 cm} \reviewersthree \hspace{0.5 mm} -- \reviewersthreeinst \par
      \vspace*{1.5cm}
      \begin{center}
        {\bf \reviewerstitle} \par
        {\bf Por} \par
        \vspace*{-1ex}
        {\emph \reviewersname} \par
        \vspace*{1cm}
        \reviewersuniversity \par
        \reviewerscenter \par
        \reviewersaddress \par
        \reviewersstatecountry \par
        \vspace*{1cm}
        \reviewersdate \par
      \end{center}
    }
  }
}

% Reviewers page signed
\newenvironment{reviewerspagesigned}{
  \pagestyle{plain}
  \vspace*{-3cm}
  \hspace{-3cm}
  \includegraphics[width=20cm,height=30cm]{reviewerssigned}
}

% Dedication
\newenvironment{dedication}{
  \begin{alwayssingle}
    \pagestyle{plain}
    \begin{center}
      \vspace*{1.5cm}
      {\LARGE }
    \end{center}
    \vspace{0.5cm}
    \begin{quote}
    \begin{center}
}
{ \end{center} \end{quote} \end{alwayssingle} }

% Acknoledgements
\newenvironment{acknowledgements}{
  \pagestyle{plain}
  \begin{alwayssingle}
    \begin{center}
      \vspace*{1.5cm}
      {\Large \bfseries Agradecimentos}
    \end{center}
    \vspace{0.5cm}
    \begin{quote}
}
{ \end{quote} \end{alwayssingle} }

% Resumo
\newenvironment{resumo}{
  \begin{alwayssingle} 
    \pagestyle{plain}
    \begin{center}
      \vspace*{1.5cm}
      {\Large \bfseries  Resumo}
    \end{center}
    \vspace{0.5cm}
    \begin{quote}
}
{ \end{quote} \end{alwayssingle} }

% Abstract
\newenvironment{abstracts}{
  \begin{alwayssingle} 
    \pagestyle{plain}
    \begin{center}
      \vspace*{1.5cm}
      {\Large \bfseries  Abstract}
    \end{center}
    \vspace{0.5cm}
    \begin{quote}
}
{ \end{quote} \end{alwayssingle} }


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Page numbers: roman+arabic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{romanpages}{
  \setcounter{page}{1}
  \renewcommand{\thepage}{\roman{page}}
}

{\newpage\renewcommand{\thepage}{\arabic{page}}\setcounter{page}{1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% New Mathematical Operators
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\DeclareMathOperator*{\argmax}{\arg\!\max}
