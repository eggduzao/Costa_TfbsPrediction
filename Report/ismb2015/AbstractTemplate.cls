% LaTeX 2e class for URSI F 2013 abstracts 
%
% Jaakko Seppänen <jaakko.seppanen@aalto.fi>
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ursiF2013}[2013/03/14 URSI F 2013 abstract class]

\LoadClass[11pt,a4paper]{article} % % a standard article document class, A4 paper, 11pt

\RequirePackage[margin=30mm]{geometry}% 3 cm page margins
\RequirePackage[T1]{fontenc}
\RequirePackage[latin1]{inputenc}
\RequirePackage{textcomp}
\RequirePackage{mathptmx} % math font Times
\RequirePackage{graphicx} % no graphics are expected, but his does not hurt
\RequirePackage{amsmath} 
\RequirePackage{cite} 
\renewcommand\citemid{:} 

\pagestyle{empty} % no page numbers
\setlength{\parindent}{0 pt} % No indent, default is 15pt.
\setlength{\parskip}{0.3\baselineskip} % paragraph skip 4mm
\renewcommand{\rmdefault}{ptm}
% Title, author(s) and affiliation(s)
\newcommand\@affiliation{\@empty}
\newcommand\affiliation[1]{\renewcommand\@affiliation{#1}}
\newcounter{affindx}
\newcommand{\aff}[2]{%
  \refstepcounter{affindx}$^{\theaffindx}$ #2\\[1ex]
  \label{aff:#1}%
}

\newcommand{\affref}[1]{$^{\ref{aff:#1}}$}

% Section and subsection headings should not appear
%\renewcommand\section{\@startsection{section}{1}{\z@}%
%                                   {-3ex \@plus -1ex \@minus -.2ex}%
%                                   {1.5ex \@plus.2ex}%
%                                   {\normalfont\bfseries}}
\renewcommand\figurename{Fig.}

\renewcommand\maketitle{\par
  \begingroup
  \newpage
  \global\@topnum\z@   % Prevents figures from going at top of page.
  \begin{center}%
    {\Large\bfseries \@title\par}%
    \vskip 1em%
    {\normalsize\lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%

    {\small \@affiliation}%
    \vskip 1.5em%
  \end{center}%
  \par
  \endgroup
}
