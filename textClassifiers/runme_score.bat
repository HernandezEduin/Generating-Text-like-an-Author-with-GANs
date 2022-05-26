::python bleu_score.py --test-author shakespeare --show-val-metrics True --show-test-metrics False
::python bleu_score.py --test-author dickens --show-val-metrics True --show-test-metrics False
::python bleu_score.py --test-author doyle --show-val-metrics True --show-test-metrics False
python bleu_score.py --test-author eliot --show-val-metrics True --show-test-metrics False
python bleu_score.py --test-author wells --show-val-metrics True --show-test-metrics False
python bleu_score.py --test-author austen --show-val-metrics True --show-test-metrics False
::
::python bleu_score.py --test-author shakespeare --show-val-metrics False --show-test-metrics True --test-filepath ../char-rnn/samples/shakespeare_test_char.txt
::python bleu_score.py --test-author dickens --show-val-metrics False --show-test-metrics True --test-filepath ../char-rnn/samples/dickens_test_char.txt
::python bleu_score.py --test-author doyle --show-val-metrics False --show-test-metrics True --test-filepath ../char-rnn/samples/doyle_test_char.txt
::python bleu_score.py --test-author eliot --show-val-metrics False --show-test-metrics True --test-filepath ../char-rnn/samples/eliot_test_char.txt
::python bleu_score.py --test-author wells --show-val-metrics False --show-test-metrics True --test-filepath ../char-rnn/samples/wells_test_char.txt
::python bleu_score.py --test-author austen --show-val-metrics False --show-test-metrics True --test-filepath ../char-rnn/samples/austen_test_char.txt
::
::python bleu_score.py --test-author shakespeare --show-val-metrics False --show-test-metrics True --test-filepath ../textGAN/samples/shakespeare_test_seqgan.txt
::python bleu_score.py --test-author dickens --show-val-metrics False --show-test-metrics True --test-filepath ../textGAN/samples/dickens_test_seqgan.txt
::python bleu_score.py --test-author doyle --show-val-metrics False --show-test-metrics True --test-filepath ../textGAN/samples/doyle_test_seqgan.txt
::python bleu_score.py --test-author eliot --show-val-metrics False --show-test-metrics True --test-filepath ../textGAN/samples/eliot_test_seqgan.txt
::python bleu_score.py --test-author wells --show-val-metrics False --show-test-metrics True --test-filepath ../textGAN/samples/wells_test_seqgan.txt
::python bleu_score.py --test-author austen --show-val-metrics False --show-test-metrics True --test-filepath ../textGAN/samples/austen_test_seqgan.txt