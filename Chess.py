import numpy as np
import pygame
import sys

#Initialize the pygame
pygame.init()

SCREEN_SIZE = (800, 600)
BLACK = (0, 0, 0)
GRAY = (125, 125, 125)
WHITE = (255, 255, 255)
# Create the screen
screen = pygame.display.set_mode(SCREEN_SIZE)

# Text Tut
myfont = pygame.font.SysFont('Comic Sans MS', 50)
textsurface = myfont.render("P", False, (0,0,0))
screen.blit(textsurface, (100,100))
# End Tut

font = pygame.font.SysFont('Comic Sans MS', 40)
PAWN = font.render("P", False, (255, 0, 0))
ROOK = font.render("R", False, (255, 0, 0))
KNIGHT = font.render("Kn", False, (255, 0, 0))
BISHOP = font.render("B", False, (255, 0, 0))
KING = font.render("K", False, (255, 0, 0))
QUEEN = font.render("Q", False, (255, 0, 0))

def Create_Board():
    board_start = (SCREEN_SIZE[0]/16, SCREEN_SIZE[1]/30)
    board_size = (SCREEN_SIZE[0] / 1.15, SCREEN_SIZE[1] / 1.1)
    board = (board_start, board_size)
    return board
    
def Board():

    Game_Board = np.zeros((8,8), dtype = np.byte)
    #Top Left = A8
    #Top Right = H8
    #Bottom Left = A1
    #Bottom Right = H1
    A = []
    B = []
    C = []
    D = []
    E = []
    F = []
    G = []
    H = []
    for x in range(8):
        A.append((BOARD_START, BOARD_TOP + SQUARE_Y * 7 - SQUARE_Y * x, SQUARE_X, SQUARE_Y))
        B.append((BOARD_START + SQUARE_X, BOARD_TOP + SQUARE_Y * 7 - SQUARE_Y * x, SQUARE_X, SQUARE_Y))
        C.append((BOARD_START + SQUARE_X * 2, BOARD_TOP + SQUARE_Y * 7 - SQUARE_Y * x, SQUARE_X, SQUARE_Y))
        D.append((BOARD_START + SQUARE_X * 3, BOARD_TOP + SQUARE_Y * 7 - SQUARE_Y * x, SQUARE_X, SQUARE_Y))
        E.append((BOARD_START + SQUARE_X * 4, BOARD_TOP + SQUARE_Y * 7 - SQUARE_Y * x, SQUARE_X, SQUARE_Y))
        F.append((BOARD_START + SQUARE_X * 5, BOARD_TOP + SQUARE_Y * 7 - SQUARE_Y * x, SQUARE_X, SQUARE_Y))
        G.append((BOARD_START + SQUARE_X * 6, BOARD_TOP + SQUARE_Y * 7 - SQUARE_Y * x, SQUARE_X, SQUARE_Y))
        H.append((BOARD_START + SQUARE_X * 7, BOARD_TOP + SQUARE_Y * 7 - SQUARE_Y * x, SQUARE_X, SQUARE_Y))
        
#    A_B = np.hstack((A,B))
#    C_D = np.hstack((C,D))
#    E_F = np.hstack((E,F))
#    G_H = np.hstack((G,H))
#    
#    A_D = np.hstack((A_B, C_D))
#    E_H = np.hstack((E_F, G_H))
#    
#    Visual_Board = np.hstack((A_D, E_H))
    
    return Game_Board, A,B,C,D,E,F,G,H
    
def Draw_Board(pieces = []):
    #Background
    pygame.draw.rect(screen, (255,0,0), (BOARD_START - 5, BOARD_TOP - 5, BOARD_X + 13, BOARD_Y + 13), 0)
    
    #Board
    pygame.draw.rect(screen, (255, 255, 255), BOARD, 0)
        
    for x in range(len(A)):
        if x % 2 == 0:
            pygame.draw.rect(screen, BLACK, A[x], 0)
        else:
            pygame.draw.rect(screen, WHITE, A[x], 0)
            
    for x in range(len(B)):
        if x % 2 != 0:
            pygame.draw.rect(screen, BLACK, B[x], 0)
        else:
            pygame.draw.rect(screen, WHITE, B[x], 0)
    for x in range(len(C)):
        if x % 2 == 0:
            pygame.draw.rect(screen, BLACK, C[x], 0)
        else:
            pygame.draw.rect(screen, WHITE, C[x], 0)
    for x in range(len(D)):
        if x % 2 != 0:
            pygame.draw.rect(screen, BLACK, D[x], 0)
        else:
            pygame.draw.rect(screen, WHITE, D[x], 0)
    for x in range(len(E)):
        if x % 2 == 0:
            pygame.draw.rect(screen, BLACK, E[x], 0)
        else:
            pygame.draw.rect(screen, WHITE, E[x], 0)
    for x in range(len(F)):
        if x % 2 != 0:
            pygame.draw.rect(screen, BLACK, F[x], 0)
        else:
            pygame.draw.rect(screen, WHITE, F[x], 0)
    for x in range(len(G)):
        if x % 2 == 0:
            pygame.draw.rect(screen, BLACK, G[x], 0)
        else:
            pygame.draw.rect(screen, WHITE, G[x], 0)
    for x in range(len(H)):
        if x % 2 != 0:
            pygame.draw.rect(screen, BLACK, H[x], 0)
        else:
            pygame.draw.rect(screen, WHITE, H[x], 0)
            
    for x in pieces:
        if x.type == 1:
            screen.blit(PAWN, Get_Area(x.Get_Location()))
        elif x.type == 2:
            screen.blit(ROOK, Get_Area(x.Get_Location()))
        elif x.type == 3:
            screen.blit(KNIGHT, Get_Area(x.Get_Location()))
        elif x.type == 4:
            screen.blit(BISHOP, Get_Area(x.Get_Location()))
        elif x.type == 5:
            screen.blit(QUEEN, Get_Area(x.Get_Location()))
        else:
            screen.blit(KING, Get_Area(x.Get_Location()))
    
def Draw_Markers(markers, places):
    for x in range(markers):
        pygame.draw.circle(screen, GRAY, (int(places[x][0] + (SQUARE_X / 2)), int(places[x][1] + (SQUARE_Y / 2))), 15)
        
def Check_Up(board, current, piece, last = False):
    if (current[0] == 0):
        return False
        
    one_up = (current[0] - 1, current[1])
    
    if board.matrix_board[one_up] != 0 and (piece.type != 3):
        return False
    
    current_piece = (one_up[0] * 8 + one_up[1])
    
    if board.matrix_board[one_up] != 0 and ((piece.type == 2 or piece.type == 5 or piece.type == 6) and board.game_board[current_piece].color != piece.color):
        piece.available_moves.append(one_up)
        return False
    
    if board.matrix_board[one_up] == 0 and (piece.type == 2 or piece.type == 5):
        piece.available_moves.append(one_up)
        Check_Up(board, one_up, piece)
        
    if board.matrix_board[one_up] == 0 and (piece.type == 1 and piece.move_counter == 0) and last == False:
        piece.available_moves.append(one_up)
        Check_Up(board, one_up, piece, True)
        
    if board.matrix_board[one_up] == 0 and (piece.type == 1 and piece.move_counter == 0) and last == True:
        piece.available_moves.append(one_up)
        return False
    
    if board.matrix_board[one_up] == 0 and (piece.type == 1 and piece.move_counter != 0) or piece.type == 6:
        piece.available_moves.append(one_up)
        return False
    
def Check_Down(board, current, piece):
    if (current[0] == 7):
        return False
    
    one_down = (current[0] + 1, current[1])
    
    if board.matrix_board[one_down] != 0 and (piece.type != 3):
        return False
    
    current_piece = (one_down[0] * 8 + one_down[1])
    
    if board.matrix_board[one_down] != 0 and ((piece.type == 2 or piece.type == 5 or piece.type == 6) and board.game_board[current_piece].color != piece.color):
        piece.available_moves.append(one_down)
        return False
    
    if board.matrix_board[one_down] == 0 and (piece.type == 2 or piece.type == 5):
        piece.available_moves.append(one_down)
        Check_Down(board, one_down, piece)
        
    if board.matrix_board[one_down] == 0 and (piece.type == 6):
        piece.available_moves.append(one_down)
        return False
        
def Check_Diagonal_Up_Left(board, current, piece):
    if (current[0] == 0 or current[1] == 0):
        return False
    
    one_diagonal = (current[0] - 1, current[1] - 1)
    
    current_piece = (one_diagonal[0] * 8 + one_diagonal[1])
    
    if board.matrix_board[one_diagonal] != 0 and piece.type == 1 and board.game_board[current_piece].color != piece.color:
        piece.available_moves.append(one_diagonal)
        return False
    
    if board.matrix_board[one_diagonal] == 0:
        piece.available_moves.append(one_diagonal)
        Check_Diagonal_Up_Left(board, one_diagonal, piece)
    
def Check_Diagonal_Up_Right(board, current, piece):
    if (current[0] == 0 or current[1] == 7):
        return False
    
    one_diagonal = (current[0] - 1, current[1] + 1)
    
    current_piece = (one_diagonal[0] * 8 + one_diagonal[1])
    
    if board.matrix_board[one_diagonal] != 0 and piece.type == 1 and board.game_board[current_piece].color != piece.color:
        piece.available_moves.append(one_diagonal)
        return False
    
    if board.matrix_board[one_diagonal] == 0:
        piece.available_moves.append(one_diagonal)
        Check_Diagonal_Up_Right(board, one_diagonal, piece)

def Check_Diagonal_Down_Left(board, current, piece):
    if (current[0] == 7 or current[1] == 0):
        return False
    
    one_diagonal = (current[0] + 1, current[1] - 1)
    
    if board.matrix_board[one_diagonal] == 0:
        piece.available_moves.append(one_diagonal)
        Check_Diagonal_Down_Left(board, one_diagonal, piece)
    
def Check_Diagonal_Down_Right(board, current, piece):
    if (current[0] == 7 or current[1] == 7):
        return False
    
    one_diagonal = (current[0] + 1, current[1] + 1)
    
    if board.matrix_board[one_diagonal] == 0:
        piece.available_moves.append(one_diagonal)
        Check_Diagonal_Down_Right(board, one_diagonal, piece)
        
#Color code - 0 is white, 1 is black
class Player():
    def __init__(self, color):
        self.color = color
        self.alive_pieces = []
        
    def Set_Board(self, board):
        self.board = board
        
    def Alive_Pieces(self):
        return self.alive_pieces
    
    def Create_Pieces(self):
        if self.color == 0:
            self.P1 = Piece(1, self.color, (6, 0))
            self.P2 = Piece(1, self.color, (6, 1))
            self.P3 = Piece(1, self.color, (6, 2))
            self.P4 = Piece(1, self.color, (6, 3))
            self.P5 = Piece(1, self.color, (6, 4))
            self.P6 = Piece(1, self.color, (6, 5))
            self.P7 = Piece(1, self.color, (6, 6))
            self.P8 = Piece(1, self.color, (6, 7))
        
            self.R1 = Piece(2, self.color, (7, 0))
            self.R2 = Piece(2, self.color, (7, 7))
            
            self.K1 = Piece(3, self.color, (7, 1))
            self.K2 = Piece(3, self.color, (7, 6))
            
            self.B1 = Piece(4, self.color, (7, 2))
            self.B2 = Piece(4, self.color, (7, 5))
            
            self.Q = Piece(5, self.color, (7, 3))
            self.K = Piece(6, self.color, (7, 4))
        else:
            self.P1 = Piece(1, self.color, (1, 0))
            self.P2 = Piece(1, self.color, (1, 1))
            self.P3 = Piece(1, self.color, (1, 2))
            self.P4 = Piece(1, self.color, (1, 3))
            self.P5 = Piece(1, self.color, (1, 4))
            self.P6 = Piece(1, self.color, (1, 5))
            self.P7 = Piece(1, self.color, (1, 6))
            self.P8 = Piece(1, self.color, (1, 7))
        
            self.R1 = Piece(2, self.color, (0, 0))
            self.R2 = Piece(2, self.color, (0, 7))
            
            self.K1 = Piece(3, self.color, (0, 1))
            self.K2 = Piece(3, self.color, (0, 6))
            
            self.B1 = Piece(4, self.color, (0, 2))
            self.B2 = Piece(4, self.color, (0, 5))
            
            self.Q = Piece(5, self.color, (0, 3))
            self.K = Piece(6, self.color, (0, 4))
            
        self.alive_pieces.append(self.P1)
        self.alive_pieces.append(self.P2)
        self.alive_pieces.append(self.P3)
        self.alive_pieces.append(self.P4)
        self.alive_pieces.append(self.P5)
        self.alive_pieces.append(self.P6)
        self.alive_pieces.append(self.P7)
        self.alive_pieces.append(self.P8)
        self.alive_pieces.append(self.R1)
        self.alive_pieces.append(self.R2)
        self.alive_pieces.append(self.K1)
        self.alive_pieces.append(self.K2)
        self.alive_pieces.append(self.B1)
        self.alive_pieces.append(self.B2)
        self.alive_pieces.append(self.Q)
        self.alive_pieces.append(self.K)
            
        
class Piece():
    def __init__(self, piece, color, location):
        #Piece will be integer from 0 to 6
        #Pawn - 1
        #Rook - 2
        #Knight - 3
        #Bishop - 4
        #Queen - 5
        #King - 6
        self.type = piece
        self.color = color
        self.isAlive = True
        self.move_counter = 0
        self.available_moves = []
        self.location = location
            
    def Set_Location(self, location):
        self.location = location
        
    def Get_Location(self):
        return self.location
    
    def Check_Alive(self):
        return self.isAlive
    
    def Death(self):
        self.isAlive = False
        
    def Draw_Moves(self):
        return len(self.available_moves), self.available_moves
            
class Game():
    def __init__(self, white, black):
        self.white = white
        self.black = black
        self.New_Game()
        self.alive = []
        
    def New_Game(self):
        self.game_board, self.A, self.B, self.C, self.D, self.E, self.F, self.G, self.H = Board()
        self.Create_Pieces()
        self.game_board, self.matrix_board = self.Set_Start_Positions()
        self.Get_Alive_Pieces()
        print(self.matrix_board)
        
    def Set_Start_Positions(self):
        temp = [self.black.R1, self.black.K1, self.black.B1, self.black.Q, self.black.K, self.black.B2, self.black.K2, self.black.R2,
                     self.black.P1, self.black.P2, self.black.P3, self.black.P4, self.black.P5, self.black.P6, self.black.P7, self.black.P8,
                     0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,
                     0,0,0,0,0,0,0,0,
                     self.white.P1, self.white.P2, self.white.P3, self.white.P4, self.white.P5, self.white.P6, self.white.P7, self.white.P8,
                     self.white.R1, self.white.K1, self.white.B1, self.white.Q, self.white.K, self.white.B2, self.white.K2, self.white.R2]
        
        matrix_temp = np.array([[2, 3, 4, 5, 6, 4, 3, 2,
                                1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1,
                                2, 3, 4, 5, 6, 4, 3, 2]], dtype = np.byte)
        matrix_temp = np.reshape(matrix_temp, (8,8))
        
        return temp, matrix_temp
    
    def Get_Board_Info(self):
        return self.Game_Board
        
    def Create_Pieces(self):
        self.white.Create_Pieces()
        self.black.Create_Pieces()
        
    def Check_Moves(self):
        for x in self.white.pieces:
            x.Check_Moves()
        for x in self.black.pieces:
            x.Check_Moves()
            
    def Select_Piece(self, piece):
        #piece = self.game_board[(location[0] * 8 + location[1])]
        marker, temp_places = piece.Draw_Moves()
        
        for x in range(marker):
            places.append(Get_Area(temp_places[x]))
        return marker, places
    
    def Deselect_Piece(self):
        marker = 0
        places = []
        return marker, places
    
    def Get_Alive_Pieces(self):
        self.alive = []
        
        for x in self.white.Alive_Pieces():
            self.alive.append(x)
        for x in self.black.Alive_Pieces():
            self.alive.append(x)
            
        return self.alive
    
    def Move_Piece(self, piece, start_location, end_location):
        if self.game_Board[end_location] == 0:
            self.game_Board[end_location] = piece
            self.game_Board[start_location] = 0
        
def Board_Grabber(mouse_event):
    x, y = mouse_event
    
    if x >= A[0][0] and x < B[0][0]:
        return 0, Y_Checker(y, A)
    
    elif x >= B[0][0] and x < C[0][0]:
        return 1, Y_Checker(y, B)
    
    elif x >= C[0][0] and x < D[0][0]:
        return 2, Y_Checker(y, C)
    
    elif x >= D[0][0] and x < E[0][0]:
        return 3, Y_Checker(y, D)
    
    elif x >= E[0][0] and x < F[0][0]:
        return 4, Y_Checker(y, E)
    
    elif x >= F[0][0] and x < G[0][0]:
        return 5, Y_Checker(y, F)
    
    elif x >= G[0][0] and x < H[0][0]:
        return 6, Y_Checker(y, G)
    
    elif x >= H[0][0] and x < (H[0][0] + H[0][2]):
        return 7, Y_Checker(y, H)
        
def Y_Checker(y, column):
    height = column[0][3]
    if y >= column[0][1] and y < (column[0][1] + height):
        return 7
    
    elif y >= column[1][1] and y < (column[1][1] + height):
        return 6
    
    elif y >= column[2][1] and y < (column[2][1] + height):
        return 5
    
    elif y >= column[3][1] and y < (column[3][1] + height):
        return 4
    
    elif y >= column[4][1] and y < (column[4][1] + height):
        return 3
    
    elif y >= column[5][1] and y < (column[5][1] + height):
        return 2
    
    elif y >= column[6][1] and y < (column[6][1] + height):
        return 1
    
    elif y >= column[7][1] and y < (column[7][1] + height):
        return 0
        
def Get_Area(places):
    if places[1] == 0:
        temp = 7 - places[0]
        return A[temp]
    elif places[1] == 1:
        temp = 7 - places[0]
        return B[temp]
    elif places[1] == 2:
        temp = 7 - places[0]
        return C[temp]
    elif places[1] == 3:
        temp = 7 - places[0]
        return D[temp]
    elif places[1] == 4:
        temp = 7 - places[0]
        return E[temp]
    elif places[1] == 5:
        temp = 7 - places[0]
        return F[temp]
    elif places[1] == 6:
        temp = 7 - places[0]
        return G[temp]
    else:
        temp = 7 - places[0]
        return H[temp]
        
running = True
BOARD = Create_Board()
BOARD_START, BOARD_TOP = BOARD[0]
BOARD_X, BOARD_Y = BOARD[1]
SQUARE_X, SQUARE_Y = BOARD_X / 8 + 1/8, BOARD_Y / 8 + 1/8

Game_Board, A,B,C,D,E,F,G,H = Board()
markers = 0
places = []
alive = []

player_1 = Player(0)
player_2 = Player(1)
current_game = Game(player_1, player_2)
alive = current_game.Get_Alive_Pieces()

Check_Up(current_game, (6,2), player_1.P3)
markers, places = current_game.Select_Piece(player_1.P3)
print(player_1.P3.available_moves)
                
while running:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            print(Board_Grabber(pygame.mouse.get_pos()))
            location = Board_Grabber(pygame.mouse.get_pos())
            
    Draw_Board(alive)
    if places != []:
        Draw_Markers(markers, places)
    pygame.display.update()