#! /usr/bin/env python

"""
See http://pydoit.org/tasks.html
"""
from doit.action import CmdAction
from doit.task import clean_targets
from doit.tools import LongRunning

import sys


def gui_open_action(pth):
    action = None
    if sys.platform.startswith('linux'):
        action = ["xdg-open", str(pth)]
    elif sys.platform.startswith('darwin'):
        action = ["open", str(pth)]
    elif sys.platform.startswith('win'):
        action = ["start", str(pth)]
    return action


def _task_html(pth, tpl_stem='html_base'):
    """
    see http://nbconvert.readthedocs.io/en/latest/usage.html
    """
    tpl_pth = 'docs/{tpl_stem}.tpl'.format(tpl_stem=tpl_stem)
    return dict(
        file_dep=[
            'docs/{pth}.ipynb'.format(pth=pth),
            tpl_pth,
        ],
        targets=[
            'docs/{pth}.html'.format(pth=pth),
            # 'docs/refs.bib'.format(pth=pth)
        ],
        actions=[
            # 'mkdir -p docs',
            # 'ln -f docs/refs.bib docs/'.format(pth=pth),
            (
                'jupyter nbconvert --to html '
                '--template=docs/{tpl_pth} '
                '--TemplateExporter.exclude_output_prompt=True '
                '--FilesWriter.build_directory=docs/ '
                # '--post=embed.EmbedPostProcessor '
                'docs/{pth}.ipynb'
            ).format(
                tpl_pth=tpl_pth,
                pth=pth
            ),
        ],
        clean=[
            # 'rm -rf docs/{pth}_files',
            clean_targets,
        ],
    )


def _task_serve_html(pth):
    """
    see https://nbconvert.readthedocs.io/en/latest/usage.html#convert-revealjs
    Quick hack to serve the damn thing without working extra syntax;
    it runs nbconvert twice tho'.
    """
    return dict(
        file_dep=[
            'docs/{pth}.slides.html'.format(pth=pth),
        ],
        targets=[
        ],
        actions=[
            # 'mkdir -p docs',
            # 'ln -f docs/refs.bib docs/'.format(pth=pth),
            LongRunning(
                'jupyter nbconvert --to slides '
                '--template=docs/{pth}_slides.tpl '
                '--TemplateExporter.exclude_output_prompt=True '
                # '--FilesWriter.build_directory=docs/ '
                # '--post=embed.EmbedPostProcessor '
                '--post=serve '
                'docs/{pth}.ipynb'.format(pth=pth)
            ),
        ],
        clean=[
            # 'rm -rf docs/{pth}_files',
            clean_targets,
        ],
    )


def _task_latex(pth, tpl_stem='latex_base'):
    """
    see http://nbconvert.readthedocs.io/en/latest/usage.html
    """
    tpl_pth = 'docs/{tpl_stem}.tplx'.format(tpl_stem=tpl_stem)
    return dict(
        file_dep=[
            'docs/{pth}.ipynb'.format(pth=pth),
            tpl_pth,
            'docs/refs.bib'.format(pth=pth),
            # 'docs/ext_media/',
        ],
        targets=[
            '_paper_output/{pth}.tex'.format(pth=pth),
            '_paper_output/refs.bib'
        ],
        actions=[
            'mkdir -p _paper_output',
            'rm -rf _paper_output/{pth}_files',
            'ln -f docs/refs.bib _paper_output'.format(pth=pth),
            'jupyter nbconvert --to latex --template={tpl_pth} '
            '--TemplateExporter.exclude_output_prompt=True '
            '--FilesWriter.build_directory=_paper_output/ '
            # '--post=embed.EmbedPostProcessor '
            'docs/{pth}.ipynb'.format(
                pth=pth,
                tpl_pth=tpl_pth,
            ),
        ],
        clean=[
            'rm -rf _paper_output/{pth}_files',
            clean_targets,
        ],
    )


def _task_pdf(pth):
    """
    """
    return dict(
        file_dep=[
            '_paper_output/refs.bib'.format(pth=pth),
            '_paper_output/{pth}.tex'.format(pth=pth)
            ],
        targets=[
            '_paper_output/{pth}.pdf'.format(pth=pth),
            '_paper_output/{pth}.aux'.format(pth=pth),
            '_paper_output/{pth}.dvi'.format(pth=pth),
            '_paper_output/{pth}.bcf'.format(pth=pth),
            '_paper_output/{pth}.blg'.format(pth=pth),
            '_paper_output/{pth}.bbl'.format(pth=pth),
            '_paper_output/{pth}.run.xml'.format(pth=pth),
            '_paper_output/texput.log',
            '_paper_output/q.log',
        ],
        actions=[
            CmdAction(
                'pdflatex -halt-on-error -interaction=batchmode '
                '{pth}'.format(pth=pth),
                cwd='_paper_output'),
            CmdAction(
                'bibtex '
                '{pth}'.format(pth=pth),
                cwd='_paper_output'),
            CmdAction(
                'pdflatex -halt-on-error -interaction=batchmode '
                '{pth}'.format(pth=pth),
                cwd='_paper_output'),
            CmdAction(
                'pdflatex -halt-on-error -interaction=batchmode '
                '{pth}'.format(pth=pth),
                cwd='_paper_output'),
        ],
        verbosity=1,
        clean=True,
    )


def _task_view_pdf(pth):
    """
    """
    return dict(
        file_dep=['_paper_output/{pth}.pdf'.format(pth=pth)],
        targets=[],
        actions=[
            gui_open_action('_paper_output/{pth}.pdf'.format(pth=pth)),
        ],
    )


def _task_share(srcpth, destpth):
    """
    """
    return dict(
        file_dep=[
            '_paper_output/{srcpth}.pdf'.format(srcpth=srcpth),
            '_paper_output/{srcpth}.tex'.format(srcpth=srcpth),
            '_paper_output/refs.bib'
        ],
        actions=[
            'mkdir -p ~/Dropbox/dan-share-stuff/tex/{destpth}/'.format(
                destpth=destpth
            ),
            CmdAction(
                'rsync -av {srcpth}_files '
                'refs.bib {srcpth}.tex '
                '{srcpth}.pdf'
                ' ~/Dropbox/dan-zdravko-stuff/tex/{destpth}/'.format(
                    srcpth=srcpth,
                    destpth=destpth
                ),
                cwd='_paper_output'
            ),
        ],
        verbosity=2
    )


def task_present_gradconf_article_deep_resynthesis():
    return _task_serve_html('gradconf_article_deep_resynthesis')


def task_html_article_deep_resynthesis():
    return _task_html('article_deep_resynthesis')


def task_latex_article_deep_resynthesis():
    return _task_latex('article_deep_resynthesis')


def task_pdf_article_deep_resynthesis():
    return _task_pdf('article_deep_resynthesis')


def task_view_pdf_article_deep_resynthesis():
    return _task_view_pdf('article_deep_resynthesis')


def task_share_article_deep_resynthesis():
    return _task_share(
        'article_deep_resynthesis',
        'learning_recursive_filters')
