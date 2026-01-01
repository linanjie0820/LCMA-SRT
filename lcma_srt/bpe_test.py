import sentencepiece as spm
import k2

# # Tokenizers
sp_asr = spm.SentencePieceProcessor(); sp_asr.load("/home/asr/ASR_AND_AST/asr4_and_ast4_imoe/data/asr4_and_ast4_bpe_all/bpe.model")
sp_st  = spm.SentencePieceProcessor(); sp_st.load("/home/asr/ASR_AND_AST/asr4_and_ast4_imoe/data/asr4_and_ast4_bpe/asr4_bpe/bpe.model")

# # Ids and vocab sizes per task
# # blank_id_asr = sp_asr.piece_to_id("<blk>")
# # print(blank_id_asr)
# # sos_id_asr = eos_id_asr = sp_asr.piece_to_id("<sos/eos>")
# # vocab_size_asr = sp_asr.get_piece_size()

# blank_id_st = sp_st.piece_to_id("<blk>")
# sos_id_st = eos_id_st = sp_st.piece_to_id("<sos/eos>")
# vocab_size_st = sp_st.get_piece_size()
text=["薄骨律镇是中国历史上南北朝时北魏北方设置的军镇之一",]
asr_tokens = sp_asr.encode(text, out_type=int)
st_tokens = sp_st.encode(text, out_type=int)
[['▁', '沸騰', 'し', 'て', 'い', 'る', '湯', 'に', '塩', 'を', '入', 'れ', 'な', 'さ', 'い']]
print(asr_tokens)
print(st_tokens)
# """
# [[6, 2, 1639, 186, 456, 3140, 2596, 2596], [6, 3, 1089, 18, 8, 97, 144, 32, 238, 303, 65, 1381, 17, 8, 11, 50, 10, 8, 2487, 83, 6, 7, 180, 932, 14, 12, 2224, 376, 69, 6, 112, 193, 128, 83, 32, 165, 357, 155, 111, 1859, 45, 13, 555, 892, 88, 156, 651, 1364]]
# """
# st_tokens = sp_st.encode(text, out_type=str)
# # print(asr_tokens)
# print(st_tokens)

# """
# [['▁', '<2zh>', '今天', '天', '气', '暖', '洋', '洋'], ['▁', '<2en>', 'bu', 't', '▁the', '▁so', 'il', '▁was', '▁mo', 'ist', '▁on', '▁reach', 'ing', '▁the', '▁to', 'p', '▁of', '▁the', '▁bank', '▁she', '▁', 's', 'li', 'pp', 'ed', '▁and', '▁fell', '▁upon', '▁her', '▁', 'k', 'ne', 'es', '▁she', '▁was', '▁st', 'ru', 'ck', '▁no', '▁doubt', '▁with', '▁a', '▁su', 'per', 'st', 'it', 'ious', '▁idea']]
# """

